import torch
import triton
import triton.language as tl
from f16s4_gemm import gemm_forward_cuda, quant_pertoken_cuda


@triton.autotune(
    configs=[
        #triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        #triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        #triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        #triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        #triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        #triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        #triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        #triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32),
    ],
    key=['n_rows', 'n_cols']
)
@triton.jit
def tokenwise_quant_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0) # 每个pid的变量是独立管理的
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0.0)
    #qrow = (row / (tl.max(tl.abs(row), axis=0) / 127) + 0.5).to(tl.int8)
    #qrow = (row / (tl.max(tl.max(row), tl.min(row)) / 127) + 0.5).to(tl.int8)
    qrow = row.to(tl.int8)
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, qrow,  mask=col_offsets<n_cols)


def tokenwise_quant(x):
    n_rows, n_cols = x.shape
    y = torch.empty((n_rows, n_cols), dtype=torch.int8, device=x.device)
    assert x.is_cuda and y.is_cuda
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    tokenwise_quant_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        #BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


@torch.no_grad()
def gemm_v1(in_f_feats, l1, l2, in_feats, kernel, scaling_factors, zeros, split_k_iters):
    in_feats = tokenwise_quant(in_f_feats)
    output = gemm_forward_cuda(in_feats, kernel, scaling_factors, zeros, split_k_iters)
    residue = l2(l1(in_f_feats))
    return output + residue
    #return output + residue


@torch.no_grad()
def gemm_v3(in_f_feats, l1, l2, in_feats, kernel, scaling_factors, zeros, split_k_iters):
    #in_feats, _ = quant_pertoken_cuda(in_f_feats)
    output = gemm_forward_cuda(in_feats, kernel, scaling_factors, zeros, split_k_iters)
    residue = l2(l1(in_f_feats))
    return output + residue


@torch.no_grad()
def gemm_v2(in_feats, l2, kernel, scaling_factors, zeros, split_k_iters):
    in_feats = tokenwise_quant(in_feats)
    output = gemm_forward_cuda(in_feats, kernel, scaling_factors, zeros, split_k_iters)
    residue, output = output[:, -128:], output[:, :-128]
    residue = residue.reshape(-1, 2, 64).sum(axis=1)
    residue = l2((residue))
    return output + residue


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],  # argument names to use as an x-axis for the plot
        x_vals=[
            1, 4, 8, 12, 16, 20, 24, 28, 32#, 64, 128, 256, 1024, 2048
        ],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'triton',
            'cuda',
            #'cuda-fused',
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Cuda",
            #"Cuda-fused",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '--')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="tokenwise-quant-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'IC': 4096, 'OC': 11008, 'G': 128, 'rank': 8},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, IC, OC, G, rank, provider):
    #x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    l1 = torch.nn.Linear(IC, rank, bias=False).to('cuda').to(torch.half)
    l1.eval()
    l2 = torch.nn.Linear(rank, OC, bias=False).to('cuda').to(torch.half)
    l2.eval()
    l22 = torch.nn.Linear(64, OC, bias=False).to('cuda').to(torch.half)
    l22.eval()
    in_feats = torch.randint(low=-128, high=127, size=(M, IC), dtype=torch.int8, device='cuda')
    in_f_feats = torch.randn(size=(M, IC), dtype=torch.float16, device='cuda')
    kernel = torch.randint(low=-128, high=127, size=(IC, OC//8), dtype=torch.int32, device='cuda')
    scaling_factors = torch.randn((IC//G, OC), dtype=torch.float16, device='cuda')
    zeros = torch.randint(low=0x01010101, high=0x01010102, size=(IC//G, OC//8), dtype=torch.int32 , device='cuda')

    kernel_1 = torch.randint(low=-128, high=127, size=(IC, (OC+128)//8), dtype=torch.int32, device='cuda')
    scaling_factors_1 = torch.randn((IC//G, OC+128), dtype=torch.float16, device='cuda')
    zeros_1 = torch.randint(low=0x01010101, high=0x01010102, size=(IC//G, (OC+128)//8), dtype=torch.int32 , device='cuda')

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gemm_v1(in_f_feats, l1, l2, in_feats, kernel, scaling_factors, zeros, 8), percentiles=quantiles, warmup=2, rep=10)
    if provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gemm_v3(in_f_feats, l1, l2, in_feats, kernel, scaling_factors, zeros, 8), percentiles=quantiles, warmup=2, rep=10)
   #if provider == 'cuda-fused':
    #    ms, min_ms, max_ms = triton.testing.do_bench(lambda: gemm_v2(in_f_feats, l22, kernel_1, scaling_factors_1, zeros_1, 8), percentiles=quantiles, warmup=2, rep=10)
    #gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    #return gbps(ms), gbps(max_ms), gbps(min_ms)
    return ms

benchmark.run(show_plots=False, print_data=True)


#if __name__ == "__main__":
#    IC = 4096
#    OC = 4096
#    G = 128
#    for M in (1,4,8,16,32,64,128,256, 256*16):
#        print(M)
#        device = torch.device("cuda")
#        in_feats = torch.randint(low=-128, high=127, size=(M, IC), dtype=torch.int8, device=device)
#        in_feats = torch.ones_like(in_feats)
#        #in_feats = torch.randn(size=(M, IC), dtype=torch.float16, device=device)
#        kernel = torch.randint(low=-8, high=7, size=(IC, OC//8), dtype=torch.int32, device=device)
#        kernel = torch.ones_like(kernel)
#        scaling_factors = torch.randn((IC//G, OC), dtype=torch.float16, device=device)
#        scaling_factors = torch.ones_like(scaling_factors)
#        zeros = torch.randint(low=0x01010101, high=0x01010102, size=(IC//G, OC//8), dtype=torch.int32 , device=device)
#        zeros = torch.zeros_like(zeros)
#        print("debug now")
#        from IPython import embed; embed()
#        # warm up
#        for i in range(10):
#            #feats, in_scales = quant_pertoken_cuda(in_feats)
#            result = gemm(in_feats, kernel, scaling_factors, zeros, 8)#8)
#
#        total_time = 0
#        start = torch.cuda.Event(enable_timing=True)
#        end = torch.cuda.Event(enable_timing=True)
#        for j in range(100):
#            start.record()
#            feats, in_scales = quant_pertoken_cuda(in_feats)
#            result = gemm(feats, kernel, scaling_factors, zeros, 8)#8)
#            end.record()
#            torch.cuda.synchronize()
#            total_time += start.elapsed_time(end)
#        print("The total time is {:.5f}".format(total_time/100))
#
