#include <torch/types.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <iostream>
#include <fstream>

#define TX 256
#define TY 1
#define BM 128
#define BN 64
#define BK 64
#define reg_m 4
#define reg_n 4
#define packed_channel 32
#define BKd16 (BK / 16)
#define reg_md4 (reg_m >> 2)
#define WARPS (TX / 32)
#define cache_per_warp 128
#define reg_nd4 (reg_n >> 2)
#define ldg_src (BN * BK / (16 * TX))
#define ldg_weight (BM * BK / (16 * TX))
#define ldg_width 16

#if (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)
extern "C"
{
    //
    // This NVVM intrinsic is subject to change in future versions of CUDA.
    // Clients should not call it directly. Rather, they should use the
    // cutlass::arch::ldsm<>() template.
    //
    __device__ uint32_t __nvvm_get_smem_pointer(void *);
}
#endif

inline __device__ unsigned get_smem_pointer(void *ptr)
{
#if (defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ >= 11)
    //
    // This NVVM intrinsic converts an address in shared memory to a plain
    // unsigned integer. This is necessary to pass to shared memory instructions
    // in inline PTX.
    //
    // In CUDA 11 and beyond, this replaces __nvvm_get_smem_pointer()  [only
    // available in 10.2].
    //
    //__device__ size_t __cvta_generic_to_shared(void* ptr);

    /// CUTLASS helper to get SMEM pointer
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));

#elif (defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ == 10 && \
       __CUDACC_VER_MINOR__ >= 2)

    return __nvvm_get_smem_pointer(ptr);

#elif defined(__CUDA_ARCH__)

    uint32_t smem_ptr;

    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
        "%0, smem_ptr; }\n"
        : "=r"(smem_ptr)
        : "l"(ptr));

    return smem_ptr;

#else

    return 0;
#endif
}

#define unpack_1(INT4)                                                                                                                            \
    INT4.w = (INT4.y & 0xF) + ((INT4.y & 0xF0) << 4) + ((INT4.y & 0xF00) << 8) + ((INT4.y & 0xF000) << 12);                                       \
    INT4.z = ((INT4.y & 0xF0000) >> 16) + ((INT4.y & 0xF00000) >> 12) + ((INT4.y & 0xF000000) >> 8) + (((INT4.y & 0xF0000000) >> 4) & 0xF000000); \
    INT4.y = (INT4.x & 0xF) + ((INT4.x & 0xF0) << 4) + ((INT4.x & 0xF00) << 8) + ((INT4.x & 0xF000) << 12);                                       \
    INT4.x = ((INT4.x & 0xF0000) >> 16) + ((INT4.x & 0xF00000) >> 12) + ((INT4.x & 0xF000000) >> 8) + (((INT4.x & 0xF0000000) >> 4) & 0xF000000);

#define unpack_2(INT4)                                                                                                                     \
    INT4.w = ((INT4.y & 0xF) << 8) + ((INT4.y & 0xF0) >> 4) + ((INT4.y & 0xF00) << 16) + ((INT4.y & 0xF000) << 4);                         \
    INT4.z = ((INT4.y & 0xF0000) >> 8) + ((INT4.y & 0xF00000) >> 20) + ((INT4.y & 0xF000000)) + (((INT4.y & 0xF0000000) >> 12) & 0xF0000); \
    INT4.y = ((INT4.x & 0xF) << 8) + ((INT4.x & 0xF0) >> 4) + ((INT4.x & 0xF00) << 16) + ((INT4.x & 0xF000) << 4);                         \
    INT4.x = ((INT4.x & 0xF0000) >> 8) + ((INT4.x & 0xF00000) >> 20) + ((INT4.x & 0xF000000)) + (((INT4.x & 0xF0000000) >> 12) & 0xF0000);

#define unpack_1_new(INT4)                                                                                                                        \
    INT4.z = (INT4.y & 0xF) + ((INT4.y & 0xF0) << 4) + ((INT4.y & 0xF00) << 8) + ((INT4.y & 0xF000) << 12);                                       \
    INT4.w = ((INT4.y & 0xF0000) >> 16) + ((INT4.y & 0xF00000) >> 12) + ((INT4.y & 0xF000000) >> 8) + (((INT4.y & 0xF0000000) >> 4) & 0xF000000); \
    INT4.y = ((INT4.x & 0xF0000) >> 16) + ((INT4.x & 0xF00000) >> 12) + ((INT4.x & 0xF000000) >> 8) + (((INT4.x & 0xF0000000) >> 4) & 0xF000000); \
    INT4.x = (INT4.x & 0xF) + ((INT4.x & 0xF0) << 4) + ((INT4.x & 0xF00) << 8) + ((INT4.x & 0xF000) << 12);

#define unpack_2_new(INT4)                                                                                                                 \
    INT4.z = ((INT4.y & 0xF) << 8) + ((INT4.y & 0xF0) >> 4) + ((INT4.y & 0xF00) << 16) + ((INT4.y & 0xF000) << 4);                         \
    INT4.w = ((INT4.y & 0xF0000) >> 8) + ((INT4.y & 0xF00000) >> 20) + ((INT4.y & 0xF000000)) + (((INT4.y & 0xF0000000) >> 12) & 0xF0000); \
    INT4.y = ((INT4.x & 0xF0000) >> 8) + ((INT4.x & 0xF00000) >> 20) + ((INT4.x & 0xF000000)) + (((INT4.x & 0xF0000000) >> 12) & 0xF0000); \
    INT4.x = ((INT4.x & 0xF) << 8) + ((INT4.x & 0xF0) >> 4) + ((INT4.x & 0xF00) << 16) + ((INT4.x & 0xF000) << 4);

#define fma_repalce(sum, x, y)  \
    asm volatile(               \
        "mul.f32 %0, %1, %2;\n" \
        : "=f"(s)               \
        : "f"(x), "f"(y));      \
    asm volatile(               \
        "add.f32 %0, %1, %2;\n" \
        : "=f"(sum)             \
        : "f"(s), "f"(sum));    \
    if (tid == -1)              \
    {                           \
        sum += s * 0;           \
    }

extern "C" __global__ void __launch_bounds__(256)
    imma_64x128(const int8_t *__restrict__ src,           //(g, K/g, N)
                const uint8_t *__restrict__ weight,       //(M, g, K/g/2)  /2 because it save int4
                const uint8_t *__restrict__ w_zeros,      //(M, g)
                const __half *__restrict__ w_base_scales, //(M, 1)
                const __half *__restrict__ w_bias_scales, //(M, 1)
                const uint8_t *__restrict__ w_bias_step,  //(M, g) int4
                const __half *__restrict__ src_scales,    //(1, N)
                __half *__restrict__ dst,
                int m, int n, int k, int g)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t bidx = blockIdx.x;
    const uint32_t bidy = blockIdx.y;

    __shared__ int32_t smem[6144]; // (64+128)*64/4*2
    int2 reg_acc_i[reg_m][reg_n];
    float2 reg_acc_f[reg_m][reg_n];
    int4 reg_src[2][reg_nd4];
    int4 reg_w[2][reg_md4];
    // use in other way, maybe use reg_ser/w
    int4 reg_src_cache;
    int4 reg_weight_cache[2];

    int8_t w_zero[2];
    __half w_base[8];
    __half w_bias[8];
    int8_t w_step[8]; // can be optimized by relayout
    __half src_scale[4];

    uint32_t gtid = (tid >> 7);
    uint32_t tid127 = (tid & 127);
    uint32_t section = (tid127 >> 1);
    uint32_t residue = ((tid127 << 4) & 31);

    bool src_gurad;
    const int8_t *__restrict__ g_src_ptr;

    uint32_t src_real_x = bidx * BN + section;
    uint32_t src_real_y = (gtid << 5) + residue;

    src_gurad = src_real_x < n && src_real_y < k;
    int src_offset = src_real_x * k + src_real_y;
    g_src_ptr = src + src_offset;

    const uint32_t section_section = (section >> 2);
    const uint32_t section_residue = (section & 3);
    const uint32_t section_factor = ((section & 15) >> 2);
    const uint32_t crosswise_offset =
        ((section_residue >> 1) << 4) +
        (((section_residue & 1) ^ (section_factor >> 1)) << 3);
    const uint32_t residue_offset = ((residue >> 4) ^ (section_factor & 1)) << 2;

    // next + 64 * BK / 8
    int32_t *write_src_s = smem + section_section * BK + crosswise_offset +
                           residue_offset + (gtid << 5);

    int iter = k >> 5; // 2 * block_iter = k  / 64 * 2, every block has tow iters to divide threads to two groups(0-127,128-255)
    const int iter_per_g = iter / g;
    uint32_t tid31 = (tid & 31);
    uint32_t warp_idx = (tid >> 5);
    uint32_t warp_strided = (warp_idx << 1);
    uint32_t htid = (tid31 >> 4);
    // const uint32_t w_strided = bidy * BM / 8 + warp_strided;
    uint32_t w_real_y = bidy * BM + (warp_idx << 4) + ((tid31 >> 1) - 8 * htid);
    uint32_t w_real_x = ((tid31 & 1) << 4) + (htid << 5);
    bool w_guard = w_real_y < m && w_real_x < k;
    // k * 8 is a stride // need to half
    const uint8_t *__restrict__ g_weight_ptr0 =
        weight + (w_real_y * k + w_real_x) / 2;
    const uint8_t *__restrict__ g_weight_ptr1 = g_weight_ptr0 + k * 4; // 4 is 8*int8/int4

    // next + BK * 2
    uint32_t q = (section_residue << 3);
    uint32_t r = section_section & 3;

    int32_t *write_w_s = smem + BN * BK / 4 + warp_strided * BK * 2 + (r & 1) * BK + q +
                         residue_offset + ((r >> 1) << 5);

    uint32_t quad_idx = (tid31 >> 2);
    uint32_t idx_in_quad = (tid & 3);
    uint32_t quad_factor = ((tid & 15) >> 2);
    uint32_t crosswise_src =
        ((idx_in_quad >> 1) << 4) + (((idx_in_quad & 1) ^ (quad_factor >> 1)) << 3);
    uint32_t crosswise_w =
        ((idx_in_quad >> 1) << 4) + (((idx_in_quad & 1)) << 3);
    uint32_t warp_x = (warp_idx >> 2);
    uint32_t warp_y = (warp_idx & 3);

    int32_t *read_src_s_0 = smem + (warp_x * 8 * BK) + (quad_idx * BK) +
                            crosswise_src + ((0 ^ (quad_factor & 1)) << 2);
    int32_t *read_src_s_1 = smem + (warp_x * 8 * BK) + (quad_idx * BK) +
                            crosswise_src + ((1 ^ (quad_factor & 1)) << 2);
    int32_t *read_w_s_0 = smem + BN / 4 * BK + (warp_y * 8 * BK) +
                          (quad_idx * BK) + crosswise_w +
                          ((0 ^ (quad_factor & 1)) << 2);
    int32_t *read_w_s_1 = smem + BN / 4 * BK + (warp_y * 8 * BK) +
                          (quad_idx * BK) + crosswise_w +
                          ((1 ^ (quad_factor & 1)) << 2);

    uint32_t o_real_x = bidx * BN + (warp_x << 5) + quad_idx;
    uint32_t o_real_y = bidy * BM + (warp_y << 5) + (idx_in_quad << 1);

#pragma unroll
    for (int i = 0; i < reg_m; i++)
    {
        src_scale[i] = src_scales[o_real_x + 8 * i];

#pragma unroll
        for (int j = 0; j < reg_n; j++)
        {
            // set acc to zero
            reg_acc_f[i][j] = float2{0, 0};
            reg_acc_i[i][j] = make_int2(0, 0);
        }
    }

    int32_t smem_switch = 3072;

    bool src_guard_iter = src_gurad;

    if (src_guard_iter)
    {
        reg_src_cache = *(reinterpret_cast<const int4 *>(g_src_ptr));
    }
    else
    {
        reg_src_cache = make_int4(0, 0, 0, 0);
    }

    if (w_guard)
    {
        *(reinterpret_cast<int2 *>(reg_weight_cache)) = *(reinterpret_cast<const int2 *>(g_weight_ptr0));
        unpack_1_new(reg_weight_cache[0]);
        *(reinterpret_cast<int2 *>(reg_weight_cache + 1)) = *(reinterpret_cast<const int2 *>(g_weight_ptr1));
        unpack_1_new(reg_weight_cache[1]);

        if (iter % iter_per_g == 0)
        {
            int cur_g = g - (iter) / iter_per_g;
            for (int i = 0; i < 2; i++)
            {
                w_zero[i] = w_zeros[cur_g * m + w_real_y + i * 8];
            }
            for (int i = 0; i < 4; i++)
            {
                *(half2 *)&(w_base[2 * i]) = *(half2 *)&(w_base_scales[o_real_y + i * 8]);
                *(half2 *)&(w_bias[2 * i]) = *(half2 *)&(w_bias_scales[o_real_y + i * 8]);
                int8_t w_s = w_bias_step[cur_g * m / 2 + o_real_y / 2 + i * 4];
                w_step[2 * i] = w_s & 0xF;
                w_step[2 * i + 1] = (w_s >> 4) & 0xF;
            }
        }

#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            int8_t *w = (int8_t *)(reg_weight_cache + i);
#pragma unroll
            for (int j = 0; j < 16; j++)
            {
                w[j] -= w_zero[i];
            }
        }
    }
    else
    {
        reg_weight_cache[0] = make_int4(0, 0, 0, 0);
        reg_weight_cache[1] = make_int4(0, 0, 0, 0);
    }

    *(reinterpret_cast<int4 *>(write_src_s)) = reg_src_cache;
    *(reinterpret_cast<int4 *>(write_w_s)) = reg_weight_cache[0];
    *(reinterpret_cast<int4 *>(write_w_s + 2 * BK)) = reg_weight_cache[1]; // BK * 8 / (int32/int8)

    __syncthreads();

    iter -= 2;

    src_real_y += BK;
    src_guard_iter = src_gurad && src_real_y < k;

    g_src_ptr += BK;
    g_weight_ptr0 += BK >> 1;
    g_weight_ptr1 += BK >> 1;

    write_src_s += smem_switch;
    write_w_s += smem_switch;

    for (; iter >= 2; iter -= 2)
    {
        if (src_guard_iter)
        {
            reg_src_cache = *(reinterpret_cast<const int4 *>(g_src_ptr));
        }
        else
        {
            reg_src_cache = make_int4(0, 0, 0, 0);
        }

        if (w_guard)
        {
            *(reinterpret_cast<int2 *>(reg_weight_cache)) = *(reinterpret_cast<const int2 *>(g_weight_ptr0));
            unpack_1_new(reg_weight_cache[0]);
            *(reinterpret_cast<int2 *>(reg_weight_cache + 1)) = *(reinterpret_cast<const int2 *>(g_weight_ptr1));
            unpack_1_new(reg_weight_cache[1]);

            int remain = iter % iter_per_g;
            if (remain == iter_per_g - 2) // need to change scales
            {
                // compute scale and clear acc
#pragma unroll
                for (int i = 0; i < reg_m; i++)
                {
                    float bias_1 = __half2float(w_base[2 * i]) + __half2float(w_bias[2 * i]) * w_step[2 * i];
                    float bias_2 = __half2float(w_base[2 * i + 1]) + __half2float(w_bias[2 * i + 1]) * w_step[2 * i + 1];
#pragma unroll
                    for (int j = 0; j < reg_n; j++)
                    {
                        float s = 0;
                        // reg_acc_f[i][j].x += reg_acc_i[i][j].x * bias_1;
                        fma_repalce(reg_acc_f[i][j].x, float(reg_acc_i[i][j].x), bias_1);
                        // reg_acc_f[i][j].y += reg_acc_i[i][j].y * bias_2;
                        fma_repalce(reg_acc_f[i][j].y, float(reg_acc_i[i][j].y), bias_2);
                        reg_acc_i[i][j] = make_int2(0, 0);
                    }
                }

                // copy
                int cur_g = g - (iter + 2) / iter_per_g;

                for (int i = 0; i < 4; i++)
                {
                    *(half2 *)&(w_base[2 * i]) = *(half2 *)&(w_base_scales[o_real_y + i * 8]);
                    *(half2 *)&(w_bias[2 * i]) = *(half2 *)&(w_bias_scales[o_real_y + i * 8]);
                    int8_t w_s = w_bias_step[cur_g * m / 2 + o_real_y / 2 + i * 4];
                    w_step[2 * i] = w_s & 0xF;
                    w_step[2 * i + 1] = (w_s >> 4) & 0xF;
                }
            }
            if (remain == 0)
            {
                // w_zero used by next iter, so it should be update earlier
                int cur_g = g - iter / iter_per_g;
                for (int i = 0; i < 2; i++)
                {
                    w_zero[i] = w_zeros[cur_g * m + w_real_y + i * 8];
                }
            }

#pragma unroll
            for (int i = 0; i < 2; i++)
            {
                int8_t *w = (int8_t *)(reg_weight_cache + i);
#pragma unroll
                for (int j = 0; j < 16; j++)
                {
                    w[j] -= w_zero[i];
                }
            }
        }
        else
        {
            reg_weight_cache[0] = make_int4(0, 0, 0, 0);
            reg_weight_cache[1] = make_int4(0, 0, 0, 0);
        }

#pragma unroll
        for (int i = 0; i < reg_nd4; ++i)
        {
            int x, y, z, w;
            unsigned addr = get_smem_pointer(read_src_s_0 + i * 8 * BK); // BK * 32 / (int_32/int_8)
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, "
                "[%4];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "r"(addr));
            reg_src[0][i] = make_int4(x, y, z, w);
        }

#pragma unroll
        for (int j = 0; j < reg_md4; ++j)
        {
            int x, y, z, w;
            unsigned addr = get_smem_pointer(read_w_s_0 + 8 * j * BK);
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, "
                "[%4];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "r"(addr));
            reg_w[0][j] = make_int4(x, y, z, w);
        }

        smem_switch = -smem_switch;

#pragma unroll
        for (int k_inner = 0; k_inner < BKd16; k_inner++)
        {
            int comp = (k_inner & 0x1);
            int load = 1 - comp;
            if (k_inner < BKd16 - 1)
            {
                int32_t *read_src_s = (k_inner & 1) ? read_src_s_0 : read_src_s_1;
                int32_t *read_w_s = (k_inner & 1) ? read_w_s_0 : read_w_s_1;
                read_src_s += 32 * ((k_inner + 1) >> 1);
                read_w_s += 32 * ((k_inner + 1) >> 1);

#pragma unroll
                for (int i = 0; i < reg_nd4; ++i)
                {
                    int x, y, z, w;
                    unsigned addr =
                        get_smem_pointer(read_src_s + i * 8 * BK); // BK*32/4
                    asm volatile(
                        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
                        "%3}, "
                        "[%4];"
                        : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                        : "r"(addr));
                    reg_src[load][i] = make_int4(x, y, z, w);
                }

#pragma unroll
                for (int j = 0; j < reg_md4; ++j)
                {
                    int x, y, z, w;
                    unsigned addr = get_smem_pointer(read_w_s + 8 * j * BK);
                    asm volatile(
                        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
                        "%3}, "
                        "[%4];"
                        : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                        : "r"(addr));
                    reg_w[load][j] = make_int4(x, y, z, w);
                }
            }

            int *B = reinterpret_cast<int *>(&reg_src[comp][0]);
            int *A = reinterpret_cast<int *>(&reg_w[comp][0]);
#pragma unroll
            for (int x = 0; x < reg_m; x++)
            {
#pragma unroll
                for (int y = 0; y < reg_n; y++)
                {
                    int *D = reinterpret_cast<int *>(&reg_acc_i[x][y]);
                    int *C = reinterpret_cast<int *>(&reg_acc_i[x][y]);

                    asm volatile(
                        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8."
                        "s32 "
                        "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                        : "=r"(D[0]), "=r"(D[1])
                        : "r"(B[y]), "r"(A[x]), "r"(C[0]), "r"(C[1]));
                }
            }
        }

        *(reinterpret_cast<int4 *>(write_src_s)) = reg_src_cache;
        *(reinterpret_cast<int4 *>(write_w_s)) = reg_weight_cache[0];
        *(reinterpret_cast<int4 *>(write_w_s + 2 * BK)) = reg_weight_cache[1]; // BK * 8 / (int32/int8)

        write_src_s += smem_switch;
        write_w_s += smem_switch;
        read_src_s_0 -= smem_switch;
        read_src_s_1 -= smem_switch;
        read_w_s_0 -= smem_switch;
        read_w_s_1 -= smem_switch;
        __syncthreads();

        src_real_y += BK;
        src_guard_iter = src_gurad && src_real_y < k;

        g_src_ptr += BK;
        g_weight_ptr0 += BK >> 1;
        g_weight_ptr1 += BK >> 1;
    }

    if (iter > 0)
    {
        if (src_guard_iter && iter > gtid)
        {
            reg_src_cache = *(reinterpret_cast<const int4 *>(g_src_ptr));
        }
        else
        {
            reg_src_cache = make_int4(0, 0, 0, 0);
        }

        if (w_guard && iter > htid)
        {
            *(reinterpret_cast<int2 *>(reg_weight_cache)) = *(reinterpret_cast<const int2 *>(g_weight_ptr0));
            unpack_1_new(reg_weight_cache[0]);
            *(reinterpret_cast<int2 *>(reg_weight_cache + 1)) = *(reinterpret_cast<const int2 *>(g_weight_ptr1));
            unpack_1_new(reg_weight_cache[1]);

            int remain = iter % iter_per_g;
            if (remain == iter_per_g - 2) // need to change scales
            {
                // compute scale and clear acc
#pragma unroll
                for (int i = 0; i < reg_m; i++)
                {
                    float bias_1 = __half2float(w_base[2 * i]) + __half2float(w_bias[2 * i]) * w_step[2 * i];
                    float bias_2 = __half2float(w_base[2 * i + 1]) + __half2float(w_bias[2 * i + 1]) * w_step[2 * i + 1];
#pragma unroll
                    for (int j = 0; j < reg_n; j++)
                    {
                        float s = 0;
                        // reg_acc_f[i][j].x += reg_acc_i[i][j].x * bias_1;
                        fma_repalce(reg_acc_f[i][j].x, float(reg_acc_i[i][j].x), bias_1);
                        // reg_acc_f[i][j].y += reg_acc_i[i][j].y * bias_2;
                        fma_repalce(reg_acc_f[i][j].y, float(reg_acc_i[i][j].y), bias_2);
                        reg_acc_i[i][j] = make_int2(0, 0);
                    }
                }

                // copy
                int cur_g = g - (iter + 2) / iter_per_g;
                for (int i = 0; i < 4; i++)
                {
                    *(half2 *)&(w_base[2 * i]) = *(half2 *)&(w_base_scales[o_real_y + i * 8]);
                    *(half2 *)&(w_bias[2 * i]) = *(half2 *)&(w_bias_scales[o_real_y + i * 8]);
                    int8_t w_s = w_bias_step[cur_g * m / 2 + o_real_y / 2 + i * 4];
                    w_step[2 * i] = w_s & 0xF;
                    w_step[2 * i + 1] = (w_s >> 4) & 0xF;
                }
            }
            if (remain == 0)
            {
                // w_zero used by next iter, so it should be update earlier
                int cur_g = g - iter / iter_per_g;
                for (int i = 0; i < 2; i++)
                {
                    w_zero[i] = w_zeros[cur_g * m + w_real_y + i * 8];
                }
            }

#pragma unroll
            for (int i = 0; i < 2; i++)
            {
                int8_t *w = (int8_t *)(reg_weight_cache + i);
#pragma unroll
                for (int j = 0; j < 16; j++)
                {
                    w[j] -= w_zero[i];
                }
            }
        }
        else
        {
            reg_weight_cache[0] = make_int4(0, 0, 0, 0);
            reg_weight_cache[1] = make_int4(0, 0, 0, 0);
        }

#pragma unroll
        for (int i = 0; i < reg_nd4; ++i)
        {
            int x, y, z, w;
            unsigned addr = get_smem_pointer(read_src_s_0 + i * 8 * BK); // BK*32/4
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, "
                "[%4];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "r"(addr));
            reg_src[0][i] = make_int4(x, y, z, w);
        }

#pragma unroll
        for (int j = 0; j < reg_md4; ++j)
        {
            int x, y, z, w;
            unsigned addr = get_smem_pointer(read_w_s_0 + 8 * j * BK);
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, "
                "[%4];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "r"(addr));
            reg_w[0][j] = make_int4(x, y, z, w);
        }

        smem_switch = -smem_switch;
#pragma unroll
        for (int k_inner = 0; k_inner < BKd16; k_inner++)
        {
            int comp = (k_inner & 0x1);
            int load = 1 - comp;
            if (k_inner < BKd16 - 1)
            {
                int32_t *read_src_s = (k_inner & 1) ? read_src_s_0 : read_src_s_1;
                int32_t *read_w_s = (k_inner & 1) ? read_w_s_0 : read_w_s_1;
                read_src_s += 32 * ((k_inner + 1) >> 1);
                read_w_s += 32 * ((k_inner + 1) >> 1);

#pragma unroll
                for (int i = 0; i < reg_nd4; ++i)
                {
                    int x, y, z, w;
                    unsigned addr =
                        get_smem_pointer(read_src_s + i * 8 * BK); // BK*32/4
                    asm volatile(
                        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
                        "%3}, "
                        "[%4];"
                        : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                        : "r"(addr));
                    reg_src[load][i] = make_int4(x, y, z, w);
                }
#pragma unroll
                for (int j = 0; j < reg_md4; ++j)
                {
                    int x, y, z, w;
                    unsigned addr = get_smem_pointer(read_w_s + 8 * j * BK);
                    asm volatile(
                        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
                        "%3}, "
                        "[%4];"
                        : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                        : "r"(addr));
                    reg_w[load][j] = make_int4(x, y, z, w);
                }
            }

            int *B = reinterpret_cast<int *>(&reg_src[comp][0]);
            int *A = reinterpret_cast<int *>(&reg_w[comp][0]);
#pragma unroll
            for (int x = 0; x < reg_m; x++)
            {
#pragma unroll
                for (int y = 0; y < reg_n; y++)
                {
                    int *D = reinterpret_cast<int *>(&reg_acc_i[x][y]);
                    int *C = reinterpret_cast<int *>(&reg_acc_i[x][y]);
                    asm volatile(
                        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8."
                        "s32 "
                        "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                        : "=r"(D[0]), "=r"(D[1])
                        : "r"(B[y]), "r"(A[x]), "r"(C[0]), "r"(C[1]));
                }
            }
        }
        *(reinterpret_cast<int4 *>(write_src_s)) = reg_src_cache;
        *(reinterpret_cast<int4 *>(write_w_s)) = reg_weight_cache[0];
        *(reinterpret_cast<int4 *>(write_w_s + 2 * BK)) = reg_weight_cache[1]; // BK * 8 / (int32/int8)

        read_src_s_0 -= smem_switch;
        read_src_s_1 -= smem_switch;
        read_w_s_0 -= smem_switch;
        read_w_s_1 -= smem_switch;

        __syncthreads();
    }

    w_guard = iter < 0;
#pragma unroll
    for (int i = 0; i < reg_nd4; ++i)
    {
        int x, y, z, w;
        unsigned addr = get_smem_pointer(read_src_s_0 + i * 8 * BK); // BK*32/4
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
            "%3}, "
            "[%4];"
            : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
            : "r"(addr));
        reg_src[0][i] = make_int4(x, y, z, w);
    }
#pragma unroll
    for (int j = 0; j < reg_md4; ++j)
    {
        int x, y, z, w;
        unsigned addr = get_smem_pointer(read_w_s_0 + 8 * j * BK);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
            "%3}, "
            "[%4];"
            : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
            : "r"(addr));
        reg_w[0][j] = make_int4(x, y, z, w);
    }

    // Special case of iter_per_g = 2, it need update
    if (iter_per_g == 2)
    {
        // compute scale and clear acc
#pragma unroll
        for (int i = 0; i < reg_m; i++)
        {
            float bias_1 = __half2float(w_base[2 * i]) + __half2float(w_bias[2 * i]) * w_step[2 * i];
            float bias_2 = __half2float(w_base[2 * i + 1]) + __half2float(w_bias[2 * i + 1]) * w_step[2 * i + 1];

#pragma unroll
            for (int j = 0; j < reg_n; j++)
            {
                float s = 0;
                // reg_acc_f[i][j].x += reg_acc_i[i][j].x * bias_1;
                fma_repalce(reg_acc_f[i][j].x, float(reg_acc_i[i][j].x), bias_1);
                // reg_acc_f[i][j].y += reg_acc_i[i][j].y * bias_2;
                fma_repalce(reg_acc_f[i][j].y, float(reg_acc_i[i][j].y), bias_2);
                reg_acc_i[i][j] = make_int2(0, 0);
            }
        }

        for (int i = 0; i < 4; i++)
        {
            int8_t w_s = w_bias_step[127 * m / 2 + o_real_y / 2 + i * 4];
            w_step[2 * i] = w_s & 0xF;
            w_step[2 * i + 1] = (w_s >> 4) & 0xF;
        }
    }
// compute
#pragma unroll
    for (int k_inner = 0; k_inner < BKd16; k_inner++)
    {
        int comp = (k_inner & 0x1);
        int load = 1 - comp;
        if (k_inner < BKd16 - 1 && !(k_inner == 1 && w_guard))
        {
            int32_t *read_src_s = (k_inner & 1) ? read_src_s_0 : read_src_s_1;
            int32_t *read_w_s = (k_inner & 1) ? read_w_s_0 : read_w_s_1;
            read_src_s += 32 * ((k_inner + 1) >> 1);
            read_w_s += 32 * ((k_inner + 1) >> 1);

#pragma unroll
            for (int i = 0; i < reg_nd4; ++i)
            {
                int x, y, z, w;
                unsigned addr = get_smem_pointer(read_src_s + i * 8 * BK); // BK*32/4
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
                    "%3}, "
                    "[%4];"
                    : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                    : "r"(addr));
                reg_src[load][i] = make_int4(x, y, z, w);
            }

#pragma unroll
            for (int j = 0; j < reg_md4; ++j)
            {
                int x, y, z, w;
                unsigned addr = get_smem_pointer(read_w_s + 8 * j * BK);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, "
                    "%3}, "
                    "[%4];"
                    : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                    : "r"(addr));
                reg_w[load][j] = make_int4(x, y, z, w);
            }
        }

        int *B = reinterpret_cast<int *>(&reg_src[comp][0]);
        int *A = reinterpret_cast<int *>(&reg_w[comp][0]);
#pragma unroll
        for (int x = 0; x < reg_m; x++)
        {
#pragma unroll
            for (int y = 0; y < reg_n; y++)
            {
                int *D = reinterpret_cast<int *>(&reg_acc_i[x][y]);
                int *C = reinterpret_cast<int *>(&reg_acc_i[x][y]);
                asm volatile(
                    "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8."
                    "s32 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=r"(D[0]), "=r"(D[1])
                    : "r"(B[y]), "r"(A[x]), "r"(C[0]), "r"(C[1]));
            }
        }
        if (k_inner == 1 && w_guard)
        {
            break;
        }
    }

    // compute scale and clear acc
#pragma unroll
    for (int i = 0; i < reg_m; i++)
    {
        float bias_1 = __half2float(w_base[2 * i]) + __half2float(w_bias[2 * i]) * w_step[2 * i];
        float bias_2 = __half2float(w_base[2 * i + 1]) + __half2float(w_bias[2 * i + 1]) * w_step[2 * i + 1];
#pragma unroll
        for (int j = 0; j < reg_n; j++)
        {
            float s = 0;
            // reg_acc_f[i][j].x += reg_acc_i[i][j].x * bias_1;
            fma_repalce(reg_acc_f[i][j].x, float(reg_acc_i[i][j].x), bias_1);
            // reg_acc_f[i][j].y += reg_acc_i[i][j].y * bias_2;
            fma_repalce(reg_acc_f[i][j].y, float(reg_acc_i[i][j].y), bias_2);
        }
    }

    __syncthreads();

    /// output
    __half *g_dst_ptr = dst + o_real_x * m + o_real_y;

#pragma unroll
    for (int x = 0; x < reg_n; x++)
    {
#pragma unroll
        for (int y = 0; y < reg_m; y++)
        {
            reg_acc_f[y][x].x = reg_acc_f[y][x].x * __half2float(src_scale[x]);
            reg_acc_f[y][x].y = reg_acc_f[y][x].y * __half2float(src_scale[x]);
            __half2 *temp = (__half2 *)(&(reg_acc_f[y][x].x));
            temp->x = __float2half(reg_acc_f[y][x].x);
            temp->y = __float2half(reg_acc_f[y][x].y);

            if (o_real_x + x * 8 < n && o_real_y + y * 8 < m)
            {
                *((__half2 *)(g_dst_ptr + x * 8 * m + y * 8)) = *temp;
            }
        }
    }
}

using namespace std;

at::Tensor quant_gemm_cuda(
    torch::Tensor qinput,
    torch::Tensor in_scales, // N, 1
    torch::Tensor qweight,
    torch::Tensor w_scales, // torch.float16
    torch::Tensor w_zeros, // torch.uint8
    torch::Tensor w_step_scales,
    torch::Tensor w_qstep,
    const size_t groups)
{
    // weight: M, g, k/g
    // input: N, K(原为, B, seqlen, K):
    // groups
    // g_steps: 最低维为oc, 按oc pack
    // 在主序(最低维)方向上进行pack.
    auto M = qweight.size(0);

    // TORCH_CHECK(qinput.dim() == 3, "should pass a qinput with shape is (batch, seqlen, hidden_size)");
    auto K = qinput.size(-1); // K = ic
    auto N = qinput.numel() / K; // N = Batch * seqlen

    int block_n = (N + BN - 1) / BN;
    int block_m = (M + BM - 1) / BM;

    dim3 grid(block_n, block_m); // (x, y, z)
    dim3 block(TX);

    at::Tensor output = at::empty({N, M}, at::dtype(at::kHalf).device(at::kCUDA));

    imma_64x128<<<grid, block>>>(
        qinput.data_ptr<int8_t>(), // N, K
        qweight.data_ptr<uint8_t>(), // M, K
        w_zeros.data_ptr<uint8_t>(), // g, oc
        reinterpret_cast<__half *>(w_scales.data_ptr<at::Half>()), // oc, 1
        reinterpret_cast<__half *>(w_step_scales.data_ptr<at::Half>()), // oc, 1
        w_qstep.data_ptr<uint8_t>(), // g, oc
        reinterpret_cast<__half *>(in_scales.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
        M,
        N,
        K,
        groups);

    return output;
}
