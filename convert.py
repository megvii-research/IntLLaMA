#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =======================================
# File Name :
# Purpose : convert fp16 models into qmodel via GPTQ
# Creation Date :
# Last Modified :
# =======================================

import os
import json
import torch
import torch.nn as nn
import transformers
import functools
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from utils import GptQuantizer, QuantLinear, RandomProjector, register_fake_quant_input_hook, Smoothor, fake_quant_input
from utils.datasets import get_wikitext2, get_c4
from awq.quantize.auto_scale import apply_scale

from evaluate_mmlu import eval_mmlu

tokenizer_replace_dict = {
    "decapoda-research/llama-7b-hf": "huggyllama/llama-7b",
    "decapoda-research/llama-13b-hf": "huggyllama/llama-13b",
    "decapoda-research/llama-30b-hf": "huggyllama/llama-30b",
    "decapoda-research/llama-65b-hf": "huggyllama/llama-65b",
}

def find_linears(module, prefix=""):
    if isinstance(module, nn.Linear):
        return {prefix: module}
    res = {}
    for name, child in module.named_children():
        res.update(find_linears(child, prefix + "." + name if prefix else name))
    return res


def loop_forward_transformer_block(block, inputs, outputs, state_registers):
    for i, sample in enumerate(inputs):
        outputs[i] = block(
            sample.unsqueeze(0),
            attention_mask=state_registers["attention_mask"],
            position_ids=state_registers["position_ids"],
        )[0]


def replace_linear2qlinear(module, quantizers, svd_deltaW, prefix=""):
    for n, m in module.named_children():
        full_name = prefix + "." + n
        if isinstance(m, nn.Linear) and full_name in quantizers:
            print("pack {}".format(full_name))
            if svd_deltaW[full_name][0] is not None:
                m.weight.data = (
                    (
                        m.weight.data.cuda().float()
                        - svd_deltaW[full_name][1].cuda()
                        @ svd_deltaW[full_name][0].cuda()
                    )
                    .to(m.weight.dtype)
                    .cpu()
                )
            qlinear = QuantLinear(
                m.in_features,
                m.out_features,
                quantizers[full_name],
                svd_deltaW[full_name],
            )
            qlinear.pack(m.weight.data.clone())
            setattr(module, n, qlinear)
        new_prefix = prefix + "." + n if prefix else n
        replace_linear2qlinear(m, quantizers, svd_deltaW, new_prefix)


def evaluation(model, dataloader, device):
    model.eval()
    model.to(device)
    dataloader = dataloader.input_ids
    nsamples = dataloader.numel() // model.config.max_sequence_length
    max_seqlen = model.config.max_sequence_length

    nlls = []
    for i in tqdm(range(nsamples)):
        input_id = dataloader[:, (i * max_seqlen) : ((i + 1) * max_seqlen)].to(device)
        with torch.no_grad():
            output = model(input_id, labels=input_id)
        neg_log_likelihood = output["loss"].float() * max_seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * max_seqlen))
    print("ppl: ", ppl.item())
    model.cpu()


@torch.no_grad()
def convert(args, model, tokenizer, device):
    # build model
    dtype = model.dtype
    model_layers = model.model.layers

    if args.eval_tiny_data:
        _, testloader = {
            "wikitext2": get_wikitext2,
            "c4": get_c4,
        }[args.eval_tiny_data](
            nsamples=args.nsamples,
            seed=0,
            tokenizer=tokenizer,
            seqlen=model.config.max_sequence_length,
        )
        hooks = register_fake_quant_input_hook(args, model)
        evaluation(model, testloader, device)
        hooks.clear()

    # build calibrator (dataset)
    with open(args.calib_data, "r") as f:
        calibrator = [torch.tensor(data) for data in json.loads(f.read())]

    # a workaroud to get input-embeds, attn_mask, position_ids
    inputs_cache = torch.zeros(
        (args.nsamples, model.config.max_sequence_length, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    state_registers = {"sample_idx": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inputs_cache[state_registers["sample_idx"]] = inp
            state_registers["sample_idx"] += 1
            state_registers["attention_mask"] = kwargs["attention_mask"]
            state_registers["position_ids"] = kwargs["position_ids"]
            raise ValueError

    # run first layer forward to get input_embeds
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model_layers[0] = Catcher(model_layers[0].to(device))
    for sample in calibrator:
        try:
            model(sample.to(device))
        except ValueError:
            pass
    model_layers[0] = model_layers[0].module.cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    ######## run convert layer-by-layer to save gpu-memory  ##########
    outputs_cache = torch.zeros_like(inputs_cache)
    gpt_quantizers = {}
    svd_deltaW = {}
    for layer_idx in range(len(model_layers)):

        def _forward_hook(module, m_inputs, m_outputs, quantizer=None):
            quantizer.calc_hessian(m_inputs[0].data, m_outputs.data)

        hook_handles = []
        layer = model_layers[layer_idx].to(device)
        named_linears = find_linears(layer)
        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(named_linears.keys())]

        for s_names in sequential:
            subset = {f"model.layers.{layer_idx}." + k: v for k, v in named_linears.items() if k in s_names}
            for n, m in subset.items():  # build quantizer for every linear
                gpt_quantizers[n] = GptQuantizer(
                    layer=m,
                    bit_width=args.bit_width,
                    groupsize=args.groupsize,
                    is_perchannel=args.is_perchannel,
                    is_symetric=args.is_symetric,
                    enable_mse=True if args.bit_width == 2 else False,
                )
                _fhook_with_name = functools.partial(
                    _forward_hook, quantizer=gpt_quantizers[n]
                )
                hook_handles.append(m.register_forward_hook(_fhook_with_name))
            # run layer forward to get the inputs/outputs of every linear module
            loop_forward_transformer_block(
                layer, inputs_cache, outputs_cache, state_registers
            )
            for h in hook_handles:
                h.remove()

            for n, m in subset.items():
                print(layer_idx, n, "Quantizing ...")
                svd_deltaW_pair = gpt_quantizers[n].fasterquant(
                    percdamp=args.percdamp,
                    enable_svd_deltaW=args.enable_svd_deltaW,
                    svd_deltaW_rank=args.svd_deltaW_rank,
                    enable_gscales_decompose=args.enable_gscales_decompose,
                    quant_type=args.quant_type,
                )
                if args.enable_svd_deltaW:
                    m.weight.data = (
                        m.weight.data.float() + svd_deltaW_pair[1] @ svd_deltaW_pair[0]
                    ).to(m.weight.dtype)
                gpt_quantizers[n].free()
                gpt_quantizers[n].cpu()
                svd_deltaW[n] = tuple(
                    [x if x is None else x.cpu() for x in svd_deltaW_pair]
                )

        # re run layer forward to get output of a layer with quantization weights
        loop_forward_transformer_block(
            layer, inputs_cache, outputs_cache, state_registers
        )

        model_layers[layer_idx] = layer.cpu()
        torch.cuda.empty_cache()
        inputs_cache, outputs_cache = outputs_cache, inputs_cache

    if args.eval_tiny_data:
        hooks = register_fake_quant_input_hook(args, model)
        evaluation(model, testloader, device)
        hooks.clear()

    return model, gpt_quantizers, svd_deltaW


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="decapoda-research/llama-7b-hf")
    parser.add_argument(
        "--nsamples", type=int, default=128, help="the number of calibration samples"
    )
    parser.add_argument(
        "--calib_data", type=str, default="", help="the path of calibration dataset"
    )
    parser.add_argument(
        "--eval_tiny_data",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4"],
        help="a tiny eval dataset for debug",
    )
    parser.add_argument("--bit_width", type=int, default=4, choices=[4, 2])
    parser.add_argument("--input_bit", type=int, default=4, choices=[8, 4, 2])
    parser.add_argument("--is_perchannel", action="store_true")
    parser.add_argument("--is_symetric", action="store_true")
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="the number of elements in a group when use group-wise quantization",
    )
    parser.add_argument(
        "--enable_svd_deltaW",
        action="store_true",
        help="a flag enable svd_deltaW",
    )
    parser.add_argument(
        "--enable_random_project",
        action="store_true",
        help="a flag enable random_project",
    )
    parser.add_argument(
        "--enable_gscales_decompose",
        action="store_true",
        help="a flag enable to depose group scales",
    )
    parser.add_argument(
        "--enable_mmlu_evaluation",
        action="store_true",
        help="a flag enable to mmlu evaluation",
    )
    parser.add_argument(
        "--svd_deltaW_rank",
        type=int,
        default=8,
        help="rank of svd deltaW",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="the directory of output.pth.tar saved",
    )
    parser.add_argument(
        "--save_fake_quant",
        action="store_true",
        help="a flag to save fake_quant checkpoint for debug",
    )
    parser.add_argument(
        "--enable_input_quant",
        action="store_true",
        help="a flag enable input quantization",
    )
    parser.add_argument(
        "--enable_smooth",
        action="store_true",
        help="a flag enable smooth quantization",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="gptq",
        choices=["rtn", "gptq"],
    )
    parser.add_argument(
        '--true_sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        "--load_awq",
        type=str,
        default="",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # show&save convert configure
    for k, v in args.__dict__.items():
        print(k + " : ", v)
    with open(os.path.join(args.output_dir, "convert_config.json"), "w") as f:
        json.dump(args.__dict__, f)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,  # speedup loading
    )
    if args.model_name in tokenizer_replace_dict:
        tokenizer_name = tokenizer_replace_dict[args.model_name]
    else:
        tokenizer_name = args.model_name
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    model.eval()
    model.config.use_cache = False

    if args.load_awq:
        print("Loading pre-computed AWQ results from", args.load_awq)
        awq_results = torch.load(args.load_awq)
        new_awq_scales = []
        for prev_op_name, layer_names, scales in awq_results["scale"]:
            if "layernorm" in prev_op_name:
                continue
            new_awq_scales.append((prev_op_name, layer_names, scales))
        apply_scale(model, new_awq_scales)

    if args.enable_random_project:
        print("Run Random Project")
        model_arch = args.model_name.split("/")[-1]
        projector = RandomProjector(
            model.config.hidden_size,
            matrix_cache="./sources/project_matrix_for_{}.npy".format(model_arch),
        )
        model.cuda()  # speedup, but also can use cpu
        model = projector.project(model)
        model.cpu()

    if args.enable_smooth:
        assert not args.load_awq, "awq and smooth can not both exist"
        print("Run Smooth Quant")
        smoothor = Smoothor(model, device, alpha=0.5, calib_mode="max")
        with open(args.calib_data, "r") as f:
            calibration_data = [torch.tensor(data) for data in json.loads(f.read())]
        smoothor.calibrate_for_act_scales(calibration_data)
        smoothor.smooth()

    model, quantizers, svd_deltaW = convert(args, model, tokenizer, device)

    if args.save_fake_quant:
        model.save_pretrained(os.path.join(args.output_dir, "fakequant_checkpoint"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "fakequant_checkpoint"))

    # evaluation
    if args.enable_mmlu_evaluation:
        print("Run mmlu evaluation")
        model.cuda()
        hooks = register_fake_quant_input_hook(args, model)
        eval_mmlu(model, args.model_name, tokenizer)
        hooks.clear()
        model.cpu()

    replace_linear2qlinear(model, quantizers, svd_deltaW)

    torch.save(
        {
            "model": model.state_dict(),
            "hyparams": {
                "model_name": args.model_name,
                "bit_width": args.bit_width,
                "is_perchannel": args.is_perchannel,
                "is_symetric": args.is_symetric,
                "groupsize": args.groupsize,
                "enable_svd_deltaW": args.enable_svd_deltaW,
                "enable_random_project": args.enable_random_project,
                "enable_gscales_decompose": args.enable_gscales_decompose,
                "svd_deltaW_rank": args.svd_deltaW_rank,
            },
        },
        os.path.join(args.output_dir, "cvt_quant_checkpoint.pth.tar"),
    )
