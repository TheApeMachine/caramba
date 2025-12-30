#include <metal_stdlib>
using namespace metal;

// Must match `LayerNormParams` in `ops.mm` (layout + types).
struct LayerNormParams {
    uint d_model;
    float eps;
    uint stride_row; // in elements
};

template <bool HAS_WEIGHT, bool HAS_BIAS>
inline void layernorm_impl(
    device const half* x,
    device const half* weight,
    device const half* bias,
    device half* out,
    constant LayerNormParams& p,
    uint tid,
    uint tg_id
) {
    constexpr uint TG = 256;
    // Apple Silicon thread execution width is currently 32 threads/simdgroup.
    // If/when this changes on other GPUs, consider making TG/NSIMD adaptive.
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD; // 8

    const uint row = tg_id;
    device const half* xr = x + row * p.stride_row;
    device half* yr = out + row * p.stride_row;

    threadgroup float tg_sum[NSIMD];
    threadgroup float tg_sumsq[NSIMD];
    threadgroup float shared_mean;
    threadgroup float shared_inv;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        sum += v;
        sumsq += v * v;
    }

    const float sg_sum = simd_sum(sum);
    const float sg_sumsq = simd_sum(sumsq);
    const bool lane0 = (tid % SIMD) == 0;
    if (lane0) {
        const uint idx = tid / SIMD;
        tg_sum[idx] = sg_sum;
        tg_sumsq[idx] = sg_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        float total2 = 0.0f;
        for (uint i = 0; i < NSIMD; ++i) {
            total += tg_sum[i];
            total2 += tg_sumsq[i];
        }
        const float inv_n = 1.0f / float(p.d_model);
        const float mean = total * inv_n;
        const float var = max(total2 * inv_n - mean * mean, 0.0f);
        shared_mean = mean;
        shared_inv = rsqrt(var + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float mean = shared_mean;
    const float inv = shared_inv;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        float y = (v - mean) * inv;
        if constexpr (HAS_WEIGHT) {
            y *= float(weight[i]);
        }
        if constexpr (HAS_BIAS) {
            y += float(bias[i]);
        }
        yr[i] = half(y);
    }
}

kernel void layernorm_fp16(
    device const half* x      [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]], // (D,)
    device const half* bias   [[ buffer(2) ]], // (D,)
    device half* out          [[ buffer(3) ]],
    constant LayerNormParams& p [[ buffer(4) ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint tg_id                [[ threadgroup_position_in_grid ]]
) {
    layernorm_impl<true, true>(x, weight, bias, out, p, tid, tg_id);
}

kernel void layernorm_weight_fp16(
    device const half* x      [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]], // (D,)
    device half* out          [[ buffer(2) ]],
    constant LayerNormParams& p [[ buffer(3) ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint tg_id                [[ threadgroup_position_in_grid ]]
) {
    layernorm_impl<true, false>(x, weight, (device const half*)nullptr, out, p, tid, tg_id);
}

kernel void layernorm_noweight_fp16(
    device const half* x      [[ buffer(0) ]],
    device half* out          [[ buffer(1) ]],
    constant LayerNormParams& p [[ buffer(2) ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint tg_id                [[ threadgroup_position_in_grid ]]
) {
    layernorm_impl<false, false>(x, (device const half*)nullptr, (device const half*)nullptr, out, p, tid, tg_id);
}

