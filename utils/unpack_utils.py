import torch


def unpack(
    qweight, 
    base_scales, 
    zeros, 
    step_scales, 
    qstep, 
    groups,
    out_features,
    in_features, 
    enable_gscales_decompose=True, 
    pack_dimension="ic"
):
    w_int = {"ic": _unpack_ic, "oc": _unpack_oc}[pack_dimension](qweight, out_features, in_features)
    w_int = w_int.view(out_features, groups, -1)
    if enable_gscales_decompose:
        quant_bit = 4
        n_pack = 2
        assert groups % n_pack == 0, "groups must an even"
        qstep_unpacked = torch.zeros(out_features, groups, 1, dtype=torch.uint8).to(base_scales.device)
        qstep = qstep.T
        for i in range(n_pack):
            qstep_unpacked[i::n_pack, :, 0] = (qstep >> (i * quant_bit)) % (1<<quant_bit)
        scales = base_scales + step_scales * qstep_unpacked
    else:
        scales = base_scales
    zeros = zeros.transpose(0, 1)
    w_float = ((w_int - zeros) * scales).view(
            out_features, in_features
        )
    return w_float.half()

def _unpack_ic(qweight, out_features, in_features):
    X, Y = out_features, in_features
    w_int = torch.zeros((X, Y), dtype=torch.int8).to(qweight.device)
    pack_bit=8
    quant_bit=4
    pack_num = pack_bit // quant_bit
    w_int = w_int.view(X, Y // pack_num, pack_num)
    for i in range(pack_num):
        w_int[:, :, i] |= (qweight >> (quant_bit * i)) & (
            (1 << quant_bit) - 1
        )
    w_int = w_int.view(X, Y)
    return w_int

def _unpack_oc(qweight, out_features, in_features):
    X, Y = out_features, in_features
    w_int = torch.zeros((X, Y), dtype=torch.int8).to(qweight.device)
    pack_bit=8
    quant_bit=4
    pack_num = pack_bit // quant_bit
    w_int = w_int.view(X // pack_num, pack_num, Y)
    for i in range(pack_num):
        w_int[:, i, :] |= (qweight >> (quant_bit * i)) & (
            (1 << quant_bit) - 1
        )
    w_int = w_int.view(X, Y)
    return w_int