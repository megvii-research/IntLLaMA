import os
import re
import sys

import torch
import torch.nn as nn
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, AutoTokenizer
from utils import QuantLinear
import json


def replace_linear2qlinear(module, prefix="", **kwargs):
    for n, m in module.named_children():
        full_name = prefix + "." + n
        if isinstance(m, nn.Linear) and "lm_head" not in n:  # hack for lm_head
            qlinear = QuantLinear(
                m.in_features,
                m.out_features,
                bit_width=kwargs.get("bit_width", None),
                groupsize=kwargs.get("groupsize", -1),
                enable_gscales_decompose=kwargs.get("enable_gscales_decompose", False),
                enable_svd_deltaW=kwargs.get("enable_svd_deltaW", False),
                svd_deltaW_rank=kwargs.get("svd_deltaW_rank", 8),
            )
            setattr(module, n, qlinear)
        new_prefix = prefix + "." + n if prefix else n
        replace_linear2qlinear(m, new_prefix, **kwargs)


def build_model(args, device_map="auto"):
    if args.weight_dtype == "llm_int8":
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=True,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    elif args.weight_dtype == "fp16":
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    elif args.weight_dtype == "intllama":
        assert os.path.exists(
            args.backbone_ckpt
        ), "loading intllama model requires checkpoint"
        ckpt = torch.load(args.backbone_ckpt)
        cvt_hyparams = ckpt["hyparams"]
        config = transformers.AutoConfig.from_pretrained(args.model_name)
        def get_non_init_llama(config):
            def skip(*args, **kwargs):
                pass
            torch.nn.init.kaiming_uniform_ = skip
            torch.nn.init.uniform_ = skip
            torch.nn.init.normal_ = skip
            transformers.modeling_utils._init_weights = False
            model = LlamaForCausalLM(config)
            return model
        model = get_non_init_llama(config)
        replace_linear2qlinear(model, **cvt_hyparams)
        model.load_state_dict(ckpt["model"])
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model = model.half()
    else:
        raise RuntimeError("unavaliable weight_dtype {}".format(args.weight_dtype))
    model.eval()
    return model


class Generator():
    def __init__(model, tokenizer, temperature, top_p, top_k, num_beams, max_new_tokens,
                 **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.kwarge = kwargs
        self.init_config()

    def init_config(self):
        self.config = GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens
            **self.kwargs,
        )

    def __call__(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=self.config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        output = self.tokenizer.decode(outputs.sequences[0])
        return output


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = build_model(args)
    if args.lora_ckpt:
        if args.weight_dtype in ["fp16", "llm_int8"]:
            model = PeftModel.from_pretrained(
                model,
                args.lora_ckpt,
                torch_dtype=torch.float16,
            )
        else:
            model = PeftQModelForCausalLM.from_pretrained(
                model,
                args.lora_ckpt,
                torch_dtype=torch.float16,
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.lora_ckpt)
        except:
            print(
                "Tokenizer not found in lora_ckpt path. Using model tokenizer instead"
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    generator = Generator(model, tokenizer, args.temperature, args.top_p, args.top_k,
                          args.num_beams, args.max_new_tokens)

    print(generator(args.prompt))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="huggyllama/llama-7b")
    parser.add_argument(
        "--weight-dtype", required=True, choices=["intllama", "llm_int8", "fp16"]
    )
    parser.add_argument(
        "--backbone-ckpt", type=str, required=True, help="the path of converted checkpoint"
    )
    parser.add_argument(
        "--lora-ckpt", type=str, default="", help="the path to lora checkpoint"
    )
    parser.add_argument(
        l"--prompt", type=str, required=True, help="The query that you want to ask LLM"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    main(args)
