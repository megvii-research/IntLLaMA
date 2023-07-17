import torch


def ceiling_div(x, y):
    return (x + y - 1) // y


def fake_quantize(data, scales, zeros, qmax):
    qdata = torch.clamp(torch.round(data / scales) + zeros, 0, qmax)
    return scales * (qdata - zeros)


def quantize(data, scales, zeros, qmax):
    qdata = torch.clamp(torch.round(data / scales) + zeros, 0, qmax)
    return qdata


def dequantize(qdata, scales, zeros):
    return scales * (qdata - zeros)


def fake_quant_input(module, x, bit=8, gs=128):
    x, = x
    axis = -1
    x_shape = x.shape
    if gs != -1:
        x = x.reshape(-1, gs)
    else:
        x = x.reshape(-1, x.shape[-1])
    min_qval, max_qval = -2**(bit-1), 2**(bit-1)-1
    min_val = torch.minimum(x.min(axis)[0], torch.zeros(x.shape[0], device=x.device, dtype=x.dtype))
    max_val = torch.maximum(x.max(axis)[0], torch.zeros(x.shape[0], device=x.device, dtype=x.dtype))
    scales = torch.maximum(abs(min_val)/(-min_qval), max_val/max_qval).unsqueeze(1)
    fake_quantized_input = torch.clamp(torch.round(x / scales), min_qval, max_qval)*scales
    fake_quantized_input = fake_quantized_input.reshape(x_shape)
    return fake_quantized_input
