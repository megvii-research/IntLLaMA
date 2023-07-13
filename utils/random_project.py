from scipy import stats
import numpy as np
import os
import torch


class RandomProjector(object):
    def __init__(self, hidden_size, matrix_cache=None):
        # it's too long for generating random-matrix directly, I recommened use cache
        if matrix_cache:
            if os.path.exists(matrix_cache):
                O = np.load(matrix_cache)
            else:
                last_index = matrix_cache[::-1].index("/")
                os.makedirs(matrix_cache[:-last_index], exist_ok=True)
                O = stats.ortho_group.rvs(hidden_size)
                np.save(matrix_cache, O)
        else:
            O = stats.ortho_group.rvs(hidden_size)
        self.O = O

    @torch.no_grad()
    def _fuse_layernorm(self, model):
        with torch.no_grad():
            model.lm_head.weight *= model.model.norm.weight[None]
            model.model.norm.weight.fill_(1)
            for layer in model.model.layers:
                layer.self_attn.q_proj.weight *= layer.input_layernorm.weight[None]
                layer.self_attn.k_proj.weight *= layer.input_layernorm.weight[None]
                layer.self_attn.v_proj.weight *= layer.input_layernorm.weight[None]
                layer.mlp.gate_proj.weight *= layer.post_attention_layernorm.weight[
                    None
                ]
                layer.mlp.up_proj.weight *= layer.post_attention_layernorm.weight[None]
                layer.input_layernorm.weight.fill_(1)
                layer.post_attention_layernorm.weight.fill_(1)
        return model

    @torch.no_grad()
    def _project(self, weight, axis):
        assert axis in (0, 1)
        O = torch.tensor(self.O, device=weight.device, dtype=torch.float32)
        if axis == 0:
            weight.copy_(O.T @ weight.float())
        else:
            weight.copy_(weight.float() @ O)

    def project(self, model):
        model = self._fuse_layernorm(model)
        self._project(model.model.embed_tokens.weight, 1)
        for layer in model.model.layers:
            self._project(layer.self_attn.q_proj.weight, 1)
            self._project(layer.self_attn.k_proj.weight, 1)
            self._project(layer.self_attn.v_proj.weight, 1)
            self._project(layer.self_attn.o_proj.weight, 0)
            self._project(layer.mlp.gate_proj.weight, 1)
            self._project(layer.mlp.up_proj.weight, 1)
            self._project(layer.mlp.down_proj.weight, 0)
        self._project(model.lm_head.weight, 1)
        return model
