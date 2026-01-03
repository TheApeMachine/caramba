#include <metal_stdlib>
using namespace metal;

// Must match `LayerNormParams` in `ops.mm` (layout + types).
struct LayerNormParams {
    uint d_model;
    float eps;
    uint stride_row; // in elements
};

// Must match `LayerNormGradWParams` in `ops.mm`.
struct LayerNormGradWParams {
    uint d_model;
    uint rows;
    uint stride_row; // in elements
};

template <bool HAS_WEIGHT, bool HAS_BIAS>
inline void layernorm_impl(
    device const half* x,
    device const half* weight,
    device const half* bias,
    device half* out,
    constant LayerNormParams& p,
    threadgroup float* tg_sum,
    threadgroup float* tg_sumsq,
    threadgroup float* shared_mean,
    threadgroup float* shared_inv,
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
        *shared_mean = mean;
        *shared_inv = rsqrt(var + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float mean = *shared_mean;
    const float inv = *shared_inv;
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

template <bool HAS_WEIGHT, bool HAS_BIAS>
inline void layernorm_impl_with_stats(
    device const half* x,
    device const half* weight,
    device const half* bias,
    device half* out,
    device half* mean_out,
    device half* inv_out,
    constant LayerNormParams& p,
    threadgroup float* tg_sum,
    threadgroup float* tg_sumsq,
    threadgroup float* shared_mean,
    threadgroup float* shared_inv,
    uint tid,
    uint tg_id
) {
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD; // 8

    const uint row = tg_id;
    device const half* xr = x + row * p.stride_row;
    device half* yr = out + row * p.stride_row;

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
        const float inv = rsqrt(var + p.eps);
        *shared_mean = mean;
        *shared_inv = inv;
        mean_out[row] = half(mean);
        inv_out[row] = half(inv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float mean = *shared_mean;
    const float inv = *shared_inv;
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
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD;

    threadgroup float tg_sum[NSIMD];
    threadgroup float tg_sumsq[NSIMD];
    threadgroup float tg_mean;
    threadgroup float tg_inv;

    layernorm_impl<true, true>(x, weight, bias, out, p, tg_sum, tg_sumsq, &tg_mean, &tg_inv, tid, tg_id);
}

kernel void layernorm_weight_fp16(
    device const half* x      [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]], // (D,)
    device half* out          [[ buffer(2) ]],
    constant LayerNormParams& p [[ buffer(3) ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint tg_id                [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD;

    threadgroup float tg_sum[NSIMD];
    threadgroup float tg_sumsq[NSIMD];
    threadgroup float tg_mean;
    threadgroup float tg_inv;

    layernorm_impl<true, false>(
        x, weight, (device const half*)nullptr, out, p, tg_sum, tg_sumsq, &tg_mean, &tg_inv, tid, tg_id);
}

kernel void layernorm_noweight_fp16(
    device const half* x      [[ buffer(0) ]],
    device half* out          [[ buffer(1) ]],
    constant LayerNormParams& p [[ buffer(2) ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint tg_id                [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD;

    threadgroup float tg_sum[NSIMD];
    threadgroup float tg_sumsq[NSIMD];
    threadgroup float tg_mean;
    threadgroup float tg_inv;

    layernorm_impl<false, false>(
        x, (device const half*)nullptr, (device const half*)nullptr, out, p, tg_sum, tg_sumsq, &tg_mean, &tg_inv, tid, tg_id);
}

kernel void layernorm_fwd_stats_fp16(
    device const half* x      [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]], // (D,)
    device const half* bias   [[ buffer(2) ]], // (D,)
    device half* out          [[ buffer(3) ]],
    device half* mean_out     [[ buffer(4) ]], // (rows,)
    device half* inv_out      [[ buffer(5) ]], // (rows,)
    constant LayerNormParams& p [[ buffer(6) ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint tg_id                [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD;

    threadgroup float tg_sum[NSIMD];
    threadgroup float tg_sumsq[NSIMD];
    threadgroup float tg_mean;
    threadgroup float tg_inv;

    layernorm_impl_with_stats<true, true>(
        x, weight, bias, out, mean_out, inv_out, p, tg_sum, tg_sumsq, &tg_mean, &tg_inv, tid, tg_id);
}

kernel void layernorm_weight_fwd_stats_fp16(
    device const half* x      [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]], // (D,)
    device half* out          [[ buffer(2) ]],
    device half* mean_out     [[ buffer(3) ]], // (rows,)
    device half* inv_out      [[ buffer(4) ]], // (rows,)
    constant LayerNormParams& p [[ buffer(5) ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint tg_id                [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD;

    threadgroup float tg_sum[NSIMD];
    threadgroup float tg_sumsq[NSIMD];
    threadgroup float tg_mean;
    threadgroup float tg_inv;

    layernorm_impl_with_stats<true, false>(
        x, weight, (device const half*)nullptr, out, mean_out, inv_out, p, tg_sum, tg_sumsq, &tg_mean, &tg_inv, tid, tg_id);
}

kernel void layernorm_noweight_fwd_stats_fp16(
    device const half* x      [[ buffer(0) ]],
    device half* out          [[ buffer(1) ]],
    device half* mean_out     [[ buffer(2) ]], // (rows,)
    device half* inv_out      [[ buffer(3) ]], // (rows,)
    constant LayerNormParams& p [[ buffer(4) ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint tg_id                [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD;

    threadgroup float tg_sum[NSIMD];
    threadgroup float tg_sumsq[NSIMD];
    threadgroup float tg_mean;
    threadgroup float tg_inv;

    layernorm_impl_with_stats<false, false>(
        x, (device const half*)nullptr, (device const half*)nullptr, out, mean_out, inv_out, p, tg_sum, tg_sumsq, &tg_mean, &tg_inv, tid, tg_id);
}

kernel void layernorm_bwd_x_fp16(
    device const half* x      [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]], // (D,)
    device const half* mean_in [[ buffer(2) ]], // (rows,)
    device const half* inv_in  [[ buffer(3) ]], // (rows,)
    device const half* grad_y  [[ buffer(4) ]],
    device half* grad_x        [[ buffer(5) ]],
    constant LayerNormParams& p [[ buffer(6) ]],
    uint tid                   [[ thread_position_in_threadgroup ]],
    uint tg_id                 [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD; // 8

    const uint row = tg_id;
    device const half* xr = x + row * p.stride_row;
    device const half* gr = grad_y + row * p.stride_row;
    device half* gx = grad_x + row * p.stride_row;

    const float mean = float(mean_in[row]);
    const float inv = float(inv_in[row]);
    const float inv_n = 1.0f / float(p.d_model);

    threadgroup float tg_s1[NSIMD];
    threadgroup float tg_s2[NSIMD];
    threadgroup float shared_s1;
    threadgroup float shared_s2;

    float s1 = 0.0f;
    float s2 = 0.0f;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        const float xhat = (v - mean) * inv;
        const float g = float(gr[i]) * float(weight[i]); // dy * gamma
        s1 += g;
        s2 += g * xhat;
    }

    const float sg1 = simd_sum(s1);
    const float sg2 = simd_sum(s2);
    const bool lane0 = (tid % SIMD) == 0;
    if (lane0) {
        const uint idx = tid / SIMD;
        tg_s1[idx] = sg1;
        tg_s2[idx] = sg2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float t1 = 0.0f;
        float t2 = 0.0f;
        for (uint i = 0; i < NSIMD; ++i) {
            t1 += tg_s1[i];
            t2 += tg_s2[i];
        }
        shared_s1 = t1;
        shared_s2 = t2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float S1 = shared_s1;
    const float S2 = shared_s2;

    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        const float xhat = (v - mean) * inv;
        const float g = float(gr[i]) * float(weight[i]); // dy * gamma
        const float dx = inv * inv_n * (float(p.d_model) * g - S1 - xhat * S2);
        gx[i] = half(dx);
    }
}

kernel void layernorm_bwd_x_noweight_fp16(
    device const half* x       [[ buffer(0) ]],
    device const half* mean_in [[ buffer(1) ]], // (rows,)
    device const half* inv_in  [[ buffer(2) ]], // (rows,)
    device const half* grad_y  [[ buffer(3) ]],
    device half* grad_x        [[ buffer(4) ]],
    constant LayerNormParams& p [[ buffer(5) ]],
    uint tid                   [[ thread_position_in_threadgroup ]],
    uint tg_id                 [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD; // 8

    const uint row = tg_id;
    device const half* xr = x + row * p.stride_row;
    device const half* gr = grad_y + row * p.stride_row;
    device half* gx = grad_x + row * p.stride_row;

    const float mean = float(mean_in[row]);
    const float inv = float(inv_in[row]);
    const float inv_n = 1.0f / float(p.d_model);

    threadgroup float tg_s1[NSIMD];
    threadgroup float tg_s2[NSIMD];
    threadgroup float shared_s1;
    threadgroup float shared_s2;

    float s1 = 0.0f;
    float s2 = 0.0f;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        const float xhat = (v - mean) * inv;
        const float g = float(gr[i]); // dy
        s1 += g;
        s2 += g * xhat;
    }

    const float sg1 = simd_sum(s1);
    const float sg2 = simd_sum(s2);
    const bool lane0 = (tid % SIMD) == 0;
    if (lane0) {
        const uint idx = tid / SIMD;
        tg_s1[idx] = sg1;
        tg_s2[idx] = sg2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float t1 = 0.0f;
        float t2 = 0.0f;
        for (uint i = 0; i < NSIMD; ++i) {
            t1 += tg_s1[i];
            t2 += tg_s2[i];
        }
        shared_s1 = t1;
        shared_s2 = t2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float S1 = shared_s1;
    const float S2 = shared_s2;

    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        const float xhat = (v - mean) * inv;
        const float g = float(gr[i]); // dy
        const float dx = inv * inv_n * (float(p.d_model) * g - S1 - xhat * S2);
        gx[i] = half(dx);
    }
}

kernel void layernorm_gradw_fp16(
    device const half* x           [[ buffer(0) ]],
    device const half* mean_in     [[ buffer(1) ]], // (rows,)
    device const half* inv_in      [[ buffer(2) ]], // (rows,)
    device const half* grad_y      [[ buffer(3) ]],
    device half* grad_w            [[ buffer(4) ]], // (d_model,)
    constant LayerNormGradWParams& p [[ buffer(5) ]],
    uint tid                       [[ thread_position_in_threadgroup ]],
    uint tg_id                     [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD; // 8

    const uint i = tg_id;
    if (i >= p.d_model) {
        return;
    }

    threadgroup float tg_sum[NSIMD];

    float acc = 0.0f;
    for (uint row = tid; row < p.rows; row += TG) {
        device const half* xr = x + row * p.stride_row;
        device const half* gr = grad_y + row * p.stride_row;
        const float mean = float(mean_in[row]);
        const float inv = float(inv_in[row]);
        const float xhat = (float(xr[i]) - mean) * inv;
        acc += float(gr[i]) * xhat;
    }

    const float sg_sum = simd_sum(acc);
    const bool lane0 = (tid % SIMD) == 0;
    if (lane0) {
        tg_sum[tid / SIMD] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint k = 0; k < NSIMD; ++k) {
            total += tg_sum[k];
        }
        grad_w[i] = half(total);
    }
}

kernel void layernorm_gradb_fp16(
    device const half* grad_y      [[ buffer(0) ]],
    device half* grad_b            [[ buffer(1) ]], // (d_model,)
    constant LayerNormGradWParams& p [[ buffer(2) ]],
    uint tid                       [[ thread_position_in_threadgroup ]],
    uint tg_id                     [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD; // 8

    const uint i = tg_id;
    if (i >= p.d_model) {
        return;
    }

    threadgroup float tg_sum[NSIMD];

    float acc = 0.0f;
    for (uint row = tid; row < p.rows; row += TG) {
        device const half* gr = grad_y + row * p.stride_row;
        acc += float(gr[i]);
    }

    const float sg_sum = simd_sum(acc);
    const bool lane0 = (tid % SIMD) == 0;
    if (lane0) {
        tg_sum[tid / SIMD] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint k = 0; k < NSIMD; ++k) {
            total += tg_sum[k];
        }
        grad_b[i] = half(total);
    }
}
