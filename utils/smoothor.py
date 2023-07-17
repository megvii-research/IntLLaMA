import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import transformers
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from utils.misc import HookManager
from tqdm import tqdm


class Smoothor(object):
    def __init__(self, model, device, alpha=0.5, calib_mode="max"):
        self.model = model
        self.device = device
        self.alpha = alpha
        print(self.alpha)
        self.sum_sample = {}
        assert calib_mode in ["max", "mean"], "only support max/mean"
        self.calib_mode = calib_mode
        print(calib_mode)

    def calibrate_for_act_scales(self, calib_data):
        act_scales = {} # x_name: x_scales

        def stat_input_hook(m, x_in, x_out, name):
            if isinstance(x_in, tuple):
                x_in = x_in[0]
            x_in = x_in.reshape(-1, x_in.shape[-1]).abs().detach()
            if self.calib_mode == "max":
                current_max = torch.max(x_in, axis=0)[0].cpu()
                if name in act_scales:
                    act_scales[name] = torch.max(act_scales[name], current_max)
                else:
                    act_scales[name] = current_max
            elif self.calib_mode == "mean": # to avoid outliers dominate
                sum_val = x_in.float().sum(axis=0).cpu()
                if name in act_scales:
                    self.sum_sample[name] += x_in.shape[0]
                    act_scales[name] += sum_val
                else:
                    self.sum_sample[name] = x_in.shape[0]
                    act_scales[name] = sum_val

        hooks = HookManager()
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=n)))

        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            for data in calib_data:
                output = self.model(data.cuda())
        hooks.clear()
        self.act_scales = act_scales
        self.model.cpu()

    def smooth(self, scales_dict=None):
        for n, m in self.model.named_modules():
            if isinstance(m, LlamaDecoderLayer):
                qkv = [m.self_attn.q_proj, m.self_attn.k_proj, m.self_attn.v_proj]
                if scales_dict:
                    scale_vo = scales_dict[n+".self_attn.o_proj"]
                    scale_updown = scales_dict[n+".mlp.down_proj"]
                    self.smooth_two_linears(m.self_attn.v_proj, m.self_attn.o_proj, scale_vo, smooth_scales=scale_vo) # a workaround
                    self.smooth_two_linears(m.mlp.up_proj, m.mlp.down_proj, scale_updown, smooth_scales=scale_updown)
                else:
                    scale_vo = self.act_scales[n+".self_attn.o_proj"]
                    scale_updown = self.act_scales[n+".mlp.down_proj"]
                    self.smooth_two_linears(m.self_attn.v_proj, m.self_attn.o_proj, act_scales=scale_vo)
                    self.smooth_two_linears(m.mlp.up_proj, m.mlp.down_proj, act_scales=scale_updown)

    def calc_smooth_scales(self, weight, a_scales):
        w_scales = weight.abs().max(axis=0)[0].float().clamp(min=1e-5)
        a_scales = a_scales.float().to(w_scales.device)
        scales = (a_scales.pow(self.alpha) / w_scales.pow(1-self.alpha)).clamp(min=1e-5)
        return scales

    @torch.no_grad()
    def smooth_ln_linears(self, ln, fc_list, act_scales, inject=True, smooth_scales=None):
        if smooth_scales is None:
            weight = torch.cat([fc.weight for fc in fc_list], axis=0)
            smooth_scales = self.calc_smooth_scales(weight, act_scales)
        if inject:
            ln.weight = nn.Parameter((ln.weight.float() / smooth_scales).to(ln.weight.dtype))
            if getattr(ln, "bias", None):
                ln.bias = nn.Parameter((ln.bias.float() / smooth_scales).to(ln.bias.dtype))
            for fc in fc_list:
                fc.weight = nn.Parameter((fc.weight.float() * smooth_scales[None, ...]).to(fc.weight.dtype))
        return smooth_scales

    @torch.no_grad()
    def smooth_two_linears(self, fc1, fc2, act_scales, inject=True, smooth_scales=None):
        if smooth_scales is None:
            smooth_scales = self.calc_smooth_scales(fc2.weight, act_scales)
        if inject:
            fc1.weight = nn.Parameter((fc1.weight.float() / smooth_scales[..., None]).to(fc1.weight.dtype))
            fc2.weight = nn.Parameter((fc2.weight.float() * smooth_scales[None, ...]).to(fc2.weight.dtype))
        return smooth_scales


if __name__ == "__main__":
    import json
    with open("./sources/c4-eos-128-calib.json", "r") as f:
        train_data = [torch.tensor(data) for data in json.loads(f.read())]#[:100]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_name = "huggyllama/llama-7b"
    #model_name = "debug_8layer_llama7b"
    #config = transformers.AutoConfig.from_pretrained(model_name)
    #config.num_hidden_layers = 8
    #non_inited_model = get_non_init_llama(config)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,  # speedup loading
    )
    model.eval()
    model.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)

    smoothor = Smoothor(model, alpha=0.75)
    smoothor.train(train_data, device)

    exit(1)
