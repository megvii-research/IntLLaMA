import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import numpy as np

from utils.quantizer_utils import quantize, dequantize, fake_quantize, ceiling_div
#from utils.unpack_utils import unpack
#from load_cuda_kernel import cuda_kernel


class Quantizer(nn.Module):
    def __init__(
        self,
        bit_width,
        groupsize,
        is_perchannel,
        is_symetric,
        enable_mse,
        device,
        shape=1,
    ):
        super(Quantizer, self).__init__()
        self.bit_width = bit_width
        self.register_buffer(
            "qmax", torch.tensor(2**self.bit_width - 1, device=device)
        )
        self.register_buffer("scales", torch.ones(shape))
        self.register_buffer("zeros", torch.zeros(shape))
        self.is_perchannel = is_perchannel
        self.is_symetric = is_symetric
        self.enable_mse = enable_mse
        self.enable_gscales_decompose = False
        self.device = device
        self.is_ready = False  # a flag indicate that quant params inited

    @torch.no_grad()
    def calc_qparams(self, data=None, groupsize=None, enable_gscales_decompose=False):
        if data is None:
            data = self.layer.weight.data.clone()  # use default data
        oc, ic = data.shape[0], data.shape[1]
        if groupsize is None:
            groupsize = self.groupsize
            groups = self.groups
        else:
            groupsize = groupsize
            groups = ic // groupsize
        assert (
            data.dim() == 2
        ), "quantizer only support the weight of linear whose dim equals to 2"
        if self.is_perchannel:
            data = data.reshape(self.out_features * groups, -1)
        else:
            data = data.reshape(
                1, -1
            )  # assume weight has only a output channel(per-tensor)
        if not self.enable_mse:
            scales, zeros = self._run_minmax_observer(data)
        else:
            scales, zeros = self._run_mse_observer(data)

        if not self.is_perchannel:  # broadcast per-tensor to per-channel
            scales, zeros = scales.repeat(oc), zeros.repeat(oc)

        shape = [
            oc,
            groups,
            1,
        ]  # 1 for reduce-dim # 1 for reduce-dim # 1 for reduce-dim
        scales, zeros = scales.reshape(shape), zeros.reshape(shape)

        # TODO: 支持自动转
        self.scales, self.zeros = scales.to(torch.float16), zeros.to(torch.float16)
        self.is_ready = True

    def fake_quantize(self, data):
        if self.is_ready:
            return fake_quantize(data, self.scales, self.zeros, self.qmax)
        else:
            raise RuntimeError(
                "before invoke fake_quantize, must get quant_params first "
            )

    def quantize(self, data):
        if self.is_ready:
            return quantize(data, self.scales, self.zeros, self.qmax)
        else:
            raise RuntimeError("before invoke quantize, must get quant_params first ")

    def decompose_group_scales(self, scales=None, pack_bit=8):
        if scales is None:
            scales = self.scales
        ## 方案B
        if True:
            # scales is [oc, groups, 1]
            scales = scales.squeeze(axis=-1)
            print("Run groups scales decompose")
            assert torch.all(scales > 0), "all quant-scales must greater than zero"
            assert (
                self.groups > 1
            ), "should use group-wise quantization(groupsize != -1) when use group-scales decompose"
            scales_max, scales_min = (
                scales.max(axis=1, keepdim=True)[0],
                scales.min(axis=1, keepdim=True)[0],
            )
            step_qmax = 2 ** (pack_bit - self.bit_width) - 1
            step = scales - scales_min  # oc, gorups
            step_scales = (scales_max - scales_min) / step_qmax  # oc, 1
            q_step = quantize(step, step_scales, 0, step_qmax)
            self.register_buffer("base_scales", scales_min[..., None].clone().detach())
            self.register_buffer("step_scales", step_scales[..., None].clone().detach())
            self.register_buffer("qstep", q_step[..., None].clone().detach())
        # 方案A
        if False:
            assert self.is_perchannel, "pow2 group scales only support perchannel quant"
            self.calc_qparams(weight, _override_groupsize=True, groupsize=-1)
            base_gscales, base_gzeros = (
                self.scales.clone().squeeze(),
                self.zeros.clone().squeeze(),
            )
            groups = (
                weight.shape[1] // self.groupsize if self.groupsize != -1 else 1
            )  # split ic into groups
            gscales_shift = torch.ones(
                (weight.shape[0], groups), device=weight.device, dtype=torch.float
            )
            best_zeros = torch.zeros(
                (weight.shape[0], groups), device=weight.device, dtype=torch.float
            )
            for i in range(groups):  # 统计所有的动态范围, 求出一个scale
                group_weight = weight[:, i * groupsize : (i + 1) * groupsize]
                gscales, gzeros = self._run_minmax_observer(
                    group_weight,
                    axis=1,
                    mode="minmax" if not self.enable_mse else "mse",
                )
                min_vals = torch.minimum(
                    group_weight.min(axis=1)[0],
                    torch.zeros(group_weight.shape[0], device=self.device),
                )
                # ceil
                ceil_shift = torch.ceil(base_gscales / gscales)
                gscales_ceil = ceil_shift * base_gscales
                gzeros_ceil = torch.round(-min_vals / gscales_ceil)
                fq_gweight_ceil = fake_quantize(
                    group_weight,
                    gscales_ceil.unsqueeze(1),
                    gzeros_ceil.unsqueeze(1),
                    self.qmax,
                )
                mse_ceil = (fq_gweight_ceil - group_weight).abs_().pow_(2.4).sum(axis=1)
                # floor
                floor_shift = torch.floor(base_gscales / gscales)
                gscales_floor = base_gscales * floor_shift
                gzeros_floor = torch.round(-min_vals / gscales_floor)
                fq_gweight_floor = fake_quantize(
                    group_weight,
                    gscales_floor.unsqueeze(1),
                    gzeros_floor.unsqueeze(1),
                    self.qmax,
                )
                mse_floor = (
                    (fq_gweight_floor - group_weight).abs_().pow_(2.4).sum(axis=1)
                )
                # norm
                # fq_gweight = fake_quantize(group_weight, gscales.unsqueeze(1), gzeros.unsqueeze(1), self.qmax)
                # mse_norm = (fq_gweight - group_weight).abs_().pow_(2.4).sum(axis=1)
                gscales_shift[:, i] = ceil_shift
                gscales_shift[:, i][mse_floor <= mse_ceil] = floor_shift[
                    mse_floor <= mse_ceil
                ]
                best_zeros[:, i] = gzeros_ceil
                best_zeros[:, i][mse_floor <= mse_ceil] = gzeros_floor[
                    mse_floor <= mse_ceil
                ]
                print("debug now")
                from IPython import embed

                embed()

    def _run_minmax_observer(self, data, axis=1):
        min_val = torch.minimum(
            data.min(axis)[0], torch.zeros(data.shape[0], device=self.device)
        )
        max_val = torch.maximum(
            data.max(axis)[0], torch.zeros(data.shape[0], device=self.device)
        )
        if self.is_symetric:
            max_absval = torch.maximum(torch.abs(min_val), max_val)
            if torch.any(min_val < 0):
                min_val[min_val < 0] = -max_absval[min_val < 0]
            scales = (max_val - min_val) / self.qmax
            zeros = torch.full_like(
                scales, (self.qmax + 1) / 2
            )  # save as uint8, 但双边极值有点偏
        else:
            scales = (max_val - min_val) / self.qmax
            zeros = torch.round(-min_val / scales)
        return scales, zeros

    def _run_mse_observer(self, data, axis=1, sample_steps=100, norm=2.4):
        min_val = torch.minimum(
            data.min(axis)[0], torch.zeros(data.shape[0], device=self.device)
        )
        max_val = torch.maximum(
            data.max(axis)[0], torch.zeros(data.shape[0], device=self.device)
        )
        best = torch.full([data.shape[0]], float("inf"), device=self.device)
        for i in range(sample_steps):
            sampled_minval = min_val * (1 - i / sample_steps)
            sampled_maxval = max_val * (1 - i / sample_steps)
            sampled_scales = (sampled_maxval - sampled_minval) / self.qmax
            if not self.is_symetric:
                sampled_zeros = torch.round(-sampled_minval / sampled_scales)
            else:
                sampled_zeros = torch.full_like(sampled_scales, (self.qmax + 1) / 2)
            fq_data = fake_quantize(
                data, sampled_scales.unsqueeze(1), sampled_zeros.unsqueeze(1), self.qmax
            )
            err = (fq_data - data).abs_().pow_(norm).sum(axis=1)
            if best[0] == float("inf"):  # initilization
                scales, zeros = sampled_scales.clone(), sampled_zeros.clone()
                best = err
            else:
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    scales[tmp] = sampled_scales[tmp]
                    zeros[tmp] = sampled_zeros[tmp]
        return scales, zeros


class GptQuantizer(Quantizer):
    """
    more details can be found in https://github.com/IST-DASLab/gptq
    """

    def __init__(
        self,
        layer,
        bit_width,
        groupsize,
        is_perchannel,
        is_symetric,
        enable_mse,
        shape=1,
    ):
        super(GptQuantizer, self).__init__(
            bit_width,
            groupsize,
            is_perchannel,
            is_symetric,
            enable_mse,
            layer.weight.device,
            shape,
        )
        self.layer = layer
        self.out_features, self.in_features = layer.out_features, layer.in_features
        self.rows, self.columns = self.out_features, self.in_features
        self.groupsize = groupsize
        self.groups = 1
        if self.groupsize != -1:
            assert (
                self.in_features % self.groupsize == 0
            ), "Recommend setting the groupsize to a value that is divisible by in_features"
            self.groups = self.in_features // self.groupsize
        self.Hessian = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0

    @torch.no_grad()
    def calc_hessian(self, xin, xout):
        assert (
            xin.dim() == 3
        ), "only support the stardand input format of LLMs is (B, L, H)"
        n_batch = xin.shape[0]
        if len(xin.shape) == 3:
            xin = xin.reshape((-1, xin.shape[-1]))  # (B, L, H) -> (BL, H)
            xin = xin.t()  # (BL, H) -> (H, BL)
        self.Hessian *= self.nsamples / (self.nsamples + n_batch)
        self.nsamples += n_batch
        xin = math.sqrt(2 / self.nsamples) * xin.float()
        self.Hessian += xin.matmul(xin.t())  # (H, BL) x (BL, H) -> (H, H)

    @torch.no_grad()
    def fasterquant(
        self,
        blocksize=128,  # 和groupsize需要对齐吗?
        percdamp=0.01,
        enable_svd_deltaW=False,
        svd_deltaW_rank=8,
        enable_gscales_decompose=False,
        quant_type="gptq",
    ):
        w_dtype = self.layer.weight.data.dtype
        weight = self.layer.weight.data.clone().float()
        H = self.Hessian
        dead_idx = torch.diag(H) == 0
        H[dead_idx, dead_idx] = 1
        weight[:, dead_idx] = 0
        # apply dampening to avoid numerical issues, but only work for a model with few billion paprameters
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.device)
        H[diag, diag] += damp
        # use Cholesky to speedup & avoid numerical issues further in large models
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        if quant_type == "rtn":
            if not self.is_ready:
                self.calc_qparams(weight)
            fq_weight = self.fake_quantize(
                weight.reshape(self.out_features, self.groups, -1)
            ).reshape(weight.shape)
            if enable_svd_deltaW:
                print("svd deltaW rank : {}".format(svd_deltaW_rank))
                delta_w = self.layer.weight.data - fq_weight
                w_weights = torch.diag(Hinv)[None, :]
                delta_w_weighted = delta_w / w_weights
                U, S, Vh = torch.linalg.svd(delta_w_weighted.float())
                svd_deltaW_B = (
                    U[:, 0:svd_deltaW_rank]
                    @ torch.diag(S)[0:svd_deltaW_rank, 0:svd_deltaW_rank]
                ).to(w_dtype)
                svd_deltaW_A = (Vh[0:svd_deltaW_rank, :] * w_weights).to(w_dtype)
            else:
                svd_deltaW_A = svd_deltaW_B = None

            self.layer.weight.data = fq_weight.to(w_dtype)

            if enable_gscales_decompose:
                self.decompose_group_scales()
                self.scales = self.base_scales + self.step_scales * self.qstep
                self.enable_gscales_decompose = True
            return (svd_deltaW_A, svd_deltaW_B)

        losses = torch.zeros_like(weight)
        fq_weight = torch.zeros_like(weight)
        scales_memory = torch.zeros(
            (self.out_features, self.groups, 1),
            dtype=torch.float16,
            device=weight.device,
        )
        zeros_memory = torch.zeros(
            (self.out_features, self.groups, 1),
            dtype=torch.float16,
            device=weight.device,
        )
        if self.groupsize != -1:
            blocksize = self.groupsize  # override blocksize
        for start_i in range(
            0, self.columns, blocksize
        ):  # loop in ic dim by blocksize step
            end_i = min(start_i + blocksize, self.columns)
            if self.groupsize != -1 and (end_i - start_i) % self.groupsize == 0:
                # calc quant params in a group
                self.calc_qparams(weight[:, start_i:end_i], groupsize=self.groupsize)
                scales_memory[:, start_i // self.groupsize, 0] = self.scales.squeeze()
                zeros_memory[:, start_i // self.groupsize, 0] = self.zeros.squeeze()
            elif not self.is_ready:
                self.calc_qparams(weight)
                scales_memory = self.scales.clone()
                zeross_memory = self.zeros.clone()
            # init block vars
            block_weight = weight[:, start_i:end_i].clone()
            block_fq_weight = torch.zeros_like(block_weight)
            block_err = torch.zeros_like(block_weight)
            block_losses = torch.zeros_like(block_weight)
            block_Hinv = Hinv[start_i:end_i, start_i:end_i]
            for j in range(end_i - start_i):  # loop every column in a block
                col_weight = block_weight[:, j]
                diag_val = block_Hinv[j, j]
                col_fq_weight = self.fake_quantize(
                    col_weight[..., None, None].to(w_dtype)
                ).flatten()
                block_fq_weight[:, j] = col_fq_weight
                block_losses[:, j] = (
                    col_weight - col_fq_weight
                ) ** 2 / diag_val**2  # d^2 becuase cholesky, eq2-left
                col_err = (col_weight - col_fq_weight) / diag_val
                block_weight[:, j:] -= col_err.unsqueeze(1).matmul(
                    block_Hinv[j, j:].unsqueeze(0)
                )  # update remaining block weights
                block_err[:, j] = col_err
            fq_weight[:, start_i:end_i] = block_fq_weight
            losses[:, start_i:end_i] = block_losses / 2  # why div 2?
            # update the remaining parameters
            weight[:, end_i:] -= block_err.matmul(Hinv[start_i:end_i, end_i:])
        torch.cuda.synchronize()

        w_dtype = self.layer.weight.data.dtype
        fq_weight = fq_weight.to(w_dtype)
        self.scales, self.zeros = scales_memory, zeros_memory
        if enable_svd_deltaW:
            print("svd deltaW rank : {}".format(svd_deltaW_rank))
            delta_w = self.layer.weight.data - fq_weight
            w_weights = torch.diag(Hinv)[None, :]
            delta_w_weighted = delta_w / w_weights
            U, S, Vh = torch.linalg.svd(delta_w_weighted.float())
            svd_deltaW_B = (
                U[:, 0:svd_deltaW_rank]
                @ torch.diag(S)[0:svd_deltaW_rank, 0:svd_deltaW_rank]
            ).to(w_dtype)
            svd_deltaW_A = (Vh[0:svd_deltaW_rank, :] * w_weights).to(w_dtype)
        else:
            svd_deltaW_A = svd_deltaW_B = None

        self.layer.weight.data = fq_weight

        if enable_gscales_decompose:
            self.decompose_group_scales()
            self.scales = self.base_scales + self.step_scales * self.qstep
            self.enable_gscales_decompose = True

        return (svd_deltaW_A, svd_deltaW_B)

    def free(self):
        del self.Hessian
        del self.layer
        torch.cuda.empty_cache()


class QuantLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        quantizer: Quantizer = None,
        svd_deltaW=(None, None),
        bit_width: int = None,
        groupsize: int = -1,
        enable_gscales_decompose=False,
        enable_svd_deltaW=False,
        svd_deltaW_rank=8,
        pack_bit=8,
        pack_dimension: str = "ic",
    ):
        super(QuantLinear, self).__init__()
        assert pack_bit == 8, "only support pack_bit = 8 now"
        self.pack_bit = pack_bit
        self.dtype = torch.uint8
        self.in_features = in_features
        self.out_features = out_features
        self.quantizer = quantizer

        if quantizer is None:
            assert bit_width, "bit_width must be passed to build qlinear"
            assert groupsize, "groupsize must be passed to build qlinear"
            self.groupsize = groupsize
            self.quant_bit = bit_width
            self.enable_gscales_decompose = enable_gscales_decompose
            self.enable_svd_deltaW = enable_svd_deltaW
            self.svd_deltaW_rank = svd_deltaW_rank
            self.groups = 1
            if self.groupsize != -1:
                assert (
                    self.in_features % self.groupsize == 0
                ), "Recommend setting the groupsize to a value that is divisible by in_features"
                self.groups = self.in_features // self.groupsize

            if self.enable_gscales_decompose:
                assert (
                    self.quant_bit == 4
                ), "only implement 4-bit in enable_pow2_group_scale"
                n_pack = 2
                assert self.groups % n_pack == 0, "groups must an even"
                self.register_buffer(
                    "scales",
                    torch.zeros(*(self.out_features, 1, 1), dtype=torch.float16),
                )
                self.register_buffer(
                    "step_scales",
                    torch.zeros(*(self.out_features, 1, 1), dtype=torch.float16),
                )
                self.register_buffer(
                    "qstep",
                    torch.zeros(
                        *(self.groups, self.out_features // n_pack), dtype=torch.uint8
                    ),
                )
                self.register_buffer(
                    "zeros",
                    torch.zeros(
                        *(self.groups, self.out_features, 1), dtype=torch.uint8
                    ), # workaround for group-wise
                )
            else:
                self.register_buffer(
                    "scales",
                    torch.zeros(
                        *(self.out_features, self.groups, 1), dtype=torch.float16
                    ),
                )
                self.register_buffer(
                    "zeros",
                    torch.zeros(
                        *(self.out_features, self.groups, 1), dtype=torch.uint8
                    ),
                )

            if self.enable_svd_deltaW:
                self.register_buffer(
                    "svd_deltaW_A",
                    torch.zeros(
                        (self.svd_deltaW_rank, self.in_features), dtype=torch.float16
                    ),
                )
                self.register_buffer(
                    "svd_deltaW_B",
                    torch.zeros(
                        (self.out_features, self.svd_deltaW_rank), dtype=torch.float16
                    ),
                )
        else:
            self.groupsize = quantizer.groupsize
            self.quant_bit = quantizer.bit_width
            self.enable_gscales_decompose = quantizer.enable_gscales_decompose
            self.groups = 1
            if self.groupsize != -1:
                assert (
                    self.in_features % self.groupsize == 0
                ), "Recommend setting the groupsize to a value that is divisible by in_features"
                self.groups = self.in_features // self.groupsize
            groups = self.groups
            if self.enable_gscales_decompose:
                self.register_buffer("scales", quantizer.base_scales.clone().detach())
                self.register_buffer(
                    "step_scales", quantizer.step_scales.clone().detach()
                )
                assert (
                    self.quant_bit == 4
                ), "only implement 4-bit in enable_pow2_group_scale"
                n_pack = 2
                assert groups % n_pack == 0, "groups must an even"
                qstep = quantizer.qstep.to(torch.uint8)
                qstep_packed = torch.zeros(
                    self.out_features // n_pack, groups, dtype=torch.uint8
                )
                for i in range(n_pack):
                    qstep_packed |= qstep[i::n_pack, :, 0] << (i * self.quant_bit)
                self.register_buffer(
                    "qstep", qstep_packed.clone().detach().T,
                )  # pack two 4bit into 8bit
                self.register_buffer(
                    "zeros", quantizer.zeros.to(torch.uint8).clone().detach().transpose(0, 1)
                )
            else:
                self.register_buffer("scales", quantizer.scales.clone().detach())
                self.register_buffer(
                    "zeros", quantizer.zeros.to(torch.uint8).clone().detach()
                )

            if svd_deltaW == (None, None):
                self.enable_svd_deltaW = False
            else:
                self.enable_svd_deltaW = True
                self.register_buffer("svd_deltaW_A", svd_deltaW[0].clone().detach())
                self.register_buffer("svd_deltaW_B", svd_deltaW[1].clone().detach())

        assert pack_dimension in [
            "ic",
            "oc",
        ], "You must choose to pack in in-channel or out-channel"
        assert pack_dimension == "ic", "You should use ic-pack for good memory layout"
        self.pack_dimension = pack_dimension

        self.groups = 1
        if self.groupsize != -1:
            assert (
                self.in_features % self.groupsize == 0
            ), "Recommend setting the groupsize to a value that is divisible by in_features"
            self.groups = self.in_features // self.groupsize
        self.bias = None  # hack for llama
        if self.pack_dimension == "ic":
            self.register_buffer(
                "qweight",
                torch.zeros(
                    (
                        self.out_features,
                        ceiling_div(self.in_features * self.quant_bit, self.pack_bit),
                    ),
                    dtype=self.dtype,
                ),
            )
        else:
            self.register_buffer(
                "qweight",
                torch.zeros(
                    (
                        ceiling_div(self.out_features * self.quant_bit, self.pack_bit),
                        self.in_features,
                    ),
                    dtype=self.dtype,
                ),
            )

    def pack(self, weight):
        weight = weight.view(self.out_features, self.groups, -1)
        self.quantizer.cuda()
        weight_uint = (
            self.quantizer.quantize(weight.cuda())
            .cpu()
            .to(self.dtype)
            .view(self.out_features, self.in_features)
        )
        weight_uint = weight_uint.contiguous()
        {"ic": self._pack_ic, "oc": self._pack_oc}[self.pack_dimension](weight_uint)
        del self.quantizer

    def _pack_ic(self, weight_uint):
        X, Y = weight_uint.shape
        pack_num = self.pack_bit // self.quant_bit
        assert [X, Y // pack_num] == list(self.qweight.shape), "packed shape mismatch!"
        weight_uint = weight_uint.view(X, Y // pack_num, pack_num)
        self.qweight *= 0
        for i in range(pack_num):
            self.qweight |= weight_uint[:, :, i] << (self.quant_bit * i)

    def _pack_oc(self, weight_uint):
        X, Y = weight_uint.shape
        pack_num = self.pack_bit // self.quant_bit
        assert [X // pack_num, Y] == list(self.qweight.shape), "packed shape mismatch!"
        weight_uint = weight_uint.view(X // pack_num, pack_num, Y)
        self.qweight *= 0
        for i in range(pack_num):
            self.qweight |= weight_uint[:, i, :] << (self.quant_bit * i)

    def unpack(self):
        self.fweight = unpack(
            self.qweight,
            self.scales,
            self.zeros,
            self.step_scales,
            self.qstep,
            self.groups,
            self.out_features,
            self.in_features
        )
        # del self.qweight

    def forward(self, x):
        # quantized_x, x_scales = cuda_kernel.tokenwise_quant(x)
        # dqx = quantized_x * x_scales.unsqueeze(2)
        # output=F.linear(dqx, self.fweight, bias=self.bias)
        output = QuantMatMul.apply(
            x,
            self.qweight,
            self.scales,
            self.zeros,
            self.step_scales,
            self.qstep,
            self.groups,
            self.fweight,
        )
        if self.enable_svd_deltaW:
            svd_output = F.linear(x, self.svd_deltaW_A)
            svd_output = F.linear(svd_output, self.svd_deltaW_B)
            output = output + svd_output
        return output


class QuantMatMul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        qweight,
        base_scales,
        zeros,
        step_scales,
        qstep,
        groups,
        fweight
    ):
        quantized_x, x_scales = cuda_kernel.tokenwise_quant(x)
        output = cuda_kernel.quant_gemm(
            quantized_x, # batch, seqlen, hidden_size
            x_scales, # batch, seqlen, 1
            qweight, # oc, packed_ic
            base_scales, # oc, 1, 1
            zeros, #g, oc, 1
            step_scales, # oc, 1, 1
            qstep, # g, packed_oc
            groups,
        )
        # ctx.save_for_backward(
        #     qweight, base_scales, zeros, step_scales, qstep, groups
        # )
        # ctx.out_features = base_scales.shape[0]
        # ctx.in_features = x.shape[-1]
        ctx.save_for_backward(fweight)
        return output.reshape(x.shape[0], x.shape[1], -1)

    @staticmethod
    def backward(ctx, grad_y):
        # qweight, base_scales, zeros, step_scales, qstep, groups = ctx.saved_tensors
        # out_features = ctx.out_features
        # in_features = ctx.in_features
        # w_half = unpack(qweight, base_scales, zeros, step_scales, qstep, groups, out_features, in_features)
        w_half, = ctx.saved_tensors
        grad_x = torch.matmul(grad_y, w_half)
        return grad_x, None, None, None, None, None, None, None, None

