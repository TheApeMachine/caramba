#include <metal_stdlib>
using namespace metal;

// Must match `RMSNormParams` in `ops.mm` (layout + types).
struct RMSNormParams {
    uint d_model;
    float eps;
    uint stride_row; // in elements
};

kernel void rmsnorm_fp16(
    device const half* x      [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]],
    device half* out          [[ buffer(2) ]],
    constant RMSNormParams& p [[ buffer(3) ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint tg_id                [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD; // 8

    const uint row = tg_id;
    device const half* xr = x + row * p.stride_row;
    device half* yr = out + row * p.stride_row;

    threadgroup float tg_sum[NSIMD];
    threadgroup float shared_inv;

    float sum = 0.0f;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        sum += v * v;
    }

    const float sg_sum = simd_sum(sum);
    const bool lane0 = (tid % SIMD) == 0;
    if (lane0) {
        tg_sum[tid / SIMD] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < NSIMD; ++i) {
            total += tg_sum[i];
        }
        const float mean = total / float(p.d_model);
        shared_inv = rsqrt(mean + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv = shared_inv;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        const float w = float(weight[i]);
        yr[i] = half(v * inv * w);
    }
}

kernel void rmsnorm_noweight_fp16(
    device const half* x      [[ buffer(0) ]],
    device half* out          [[ buffer(1) ]],
    constant RMSNormParams& p [[ buffer(2) ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint tg_id                [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD; // 8

    const uint row = tg_id;
    device const half* xr = x + row * p.stride_row;
    device half* yr = out + row * p.stride_row;

    threadgroup float tg_sum[NSIMD];
    threadgroup float shared_inv;

    float sum = 0.0f;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        sum += v * v;
    }

    const float sg_sum = simd_sum(sum);
    const bool lane0 = (tid % SIMD) == 0;
    if (lane0) {
        tg_sum[tid / SIMD] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < NSIMD; ++i) {
            total += tg_sum[i];
        }
        const float mean = total / float(p.d_model);
        shared_inv = rsqrt(mean + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv = shared_inv;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        yr[i] = half(v * inv);
    }
}

