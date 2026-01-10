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

constant uint TG = 256;
constant uint SIMD = 32;
constant uint NSIMD = TG / SIMD; // 8

template <typename T, bool HAS_WEIGHT, bool HAS_BIAS>
inline void layernorm_impl(
    device const T* x,
    device const T* weight,
    device const T* bias,
    device T* out,
    constant LayerNormParams& p,
    threadgroup float* tg_sum,
    threadgroup float* tg_sumsq,
    threadgroup float* shared_mean,
    threadgroup float* shared_inv,
    uint tid,
    uint tg_id
) {
    const uint row = tg_id;
    device const T* xr = x + row * p.stride_row;
    device T* yr = out + row * p.stride_row;

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
        yr[i] = T(y);
    }
}

template <typename T, bool HAS_WEIGHT, bool HAS_BIAS>
inline void layernorm_impl_with_stats(
    device const T* x,
    device const T* weight,
    device const T* bias,
    device T* out,
    device T* mean_out,
    device T* inv_out,
    constant LayerNormParams& p,
    threadgroup float* tg_sum,
    threadgroup float* tg_sumsq,
    threadgroup float* shared_mean,
    threadgroup float* shared_inv,
    uint tid,
    uint tg_id
) {
    const uint row = tg_id;
    device const T* xr = x + row * p.stride_row;
    device T* yr = out + row * p.stride_row;

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
        mean_out[row] = T(mean);
        inv_out[row] = T(inv);
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
        yr[i] = T(y);
    }
}

// Instantiate FP16 kernels
kernel void layernorm_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]],
    device const half* bias [[ buffer(2) ]],
    device half* out [[ buffer(3) ]],
    constant LayerNormParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float ts[8], tq[8], tm, ti;
    layernorm_impl<half, true, true>(x, weight, bias, out, p, ts, tq, &tm, &ti, tid, tg_id);
}

kernel void layernorm_weight_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]],
    device half* out [[ buffer(2) ]],
    constant LayerNormParams& p [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float ts[8], tq[8], tm, ti;
    layernorm_impl<half, true, false>(x, weight, nullptr, out, p, ts, tq, &tm, &ti, tid, tg_id);
}

kernel void layernorm_noweight_fp16(
    device const half* x [[ buffer(0) ]],
    device half* out [[ buffer(1) ]],
    constant LayerNormParams& p [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float ts[8], tq[8], tm, ti;
    layernorm_impl<half, false, false>(x, nullptr, nullptr, out, p, ts, tq, &tm, &ti, tid, tg_id);
}

// ... Fwd stats FP16 ...
kernel void layernorm_fwd_stats_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]],
    device const half* bias [[ buffer(2) ]],
    device half* out [[ buffer(3) ]],
    device half* mean_out [[ buffer(4) ]],
    device half* inv_out [[ buffer(5) ]],
    constant LayerNormParams& p [[ buffer(6) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float ts[8], tq[8], tm, ti;
    layernorm_impl_with_stats<half, true, true>(x, weight, bias, out, mean_out, inv_out, p, ts, tq, &tm, &ti, tid, tg_id);
}

kernel void layernorm_weight_fwd_stats_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]],
    device half* out [[ buffer(2) ]],
    device half* mean_out [[ buffer(3) ]],
    device half* inv_out [[ buffer(4) ]],
    constant LayerNormParams& p [[ buffer(5) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float ts[8], tq[8], tm, ti;
    layernorm_impl_with_stats<half, true, false>(x, weight, nullptr, out, mean_out, inv_out, p, ts, tq, &tm, &ti, tid, tg_id);
}

kernel void layernorm_noweight_fwd_stats_fp16(
    device const half* x [[ buffer(0) ]],
    device half* out [[ buffer(1) ]],
    device half* mean_out [[ buffer(2) ]],
    device half* inv_out [[ buffer(3) ]],
    constant LayerNormParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float ts[8], tq[8], tm, ti;
    layernorm_impl_with_stats<half, false, false>(x, nullptr, nullptr, out, mean_out, inv_out, p, ts, tq, &tm, &ti, tid, tg_id);
}

// ------ FP32 Instantiations ------

kernel void layernorm_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* weight [[ buffer(1) ]],
    device const float* bias [[ buffer(2) ]],
    device float* out [[ buffer(3) ]],
    constant LayerNormParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float ts[8], tq[8], tm, ti;
    layernorm_impl<float, true, true>(x, weight, bias, out, p, ts, tq, &tm, &ti, tid, tg_id);
}

kernel void layernorm_weight_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* weight [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    constant LayerNormParams& p [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float ts[8], tq[8], tm, ti;
    layernorm_impl<float, true, false>(x, weight, nullptr, out, p, ts, tq, &tm, &ti, tid, tg_id);
}

kernel void layernorm_noweight_fp32(
    device const float* x [[ buffer(0) ]],
    device float* out [[ buffer(1) ]],
    constant LayerNormParams& p [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float ts[8], tq[8], tm, ti;
    layernorm_impl<float, false, false>(x, nullptr, nullptr, out, p, ts, tq, &tm, &ti, tid, tg_id);
}

kernel void layernorm_fwd_stats_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* weight [[ buffer(1) ]],
    device const float* bias [[ buffer(2) ]],
    device float* out [[ buffer(3) ]],
    device float* mean_out [[ buffer(4) ]],
    device float* inv_out [[ buffer(5) ]],
    constant LayerNormParams& p [[ buffer(6) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float ts[8], tq[8], tm, ti;
    layernorm_impl_with_stats<float, true, true>(x, weight, bias, out, mean_out, inv_out, p, ts, tq, &tm, &ti, tid, tg_id);
}

kernel void layernorm_weight_fwd_stats_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* weight [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    device float* mean_out [[ buffer(3) ]],
    device float* inv_out [[ buffer(4) ]],
    constant LayerNormParams& p [[ buffer(5) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float ts[8], tq[8], tm, ti;
    layernorm_impl_with_stats<float, true, false>(x, weight, nullptr, out, mean_out, inv_out, p, ts, tq, &tm, &ti, tid, tg_id);
}

kernel void layernorm_noweight_fwd_stats_fp32(
    device const float* x [[ buffer(0) ]],
    device float* out [[ buffer(1) ]],
    device float* mean_out [[ buffer(2) ]],
    device float* inv_out [[ buffer(3) ]],
    constant LayerNormParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float ts[8], tq[8], tm, ti;
    layernorm_impl_with_stats<float, false, false>(x, nullptr, nullptr, out, mean_out, inv_out, p, ts, tq, &tm, &ti, tid, tg_id);
}


// ------ Backward Kernels Templated ------

template <typename T>
inline void layernorm_bwd_x_impl(
    device const T* x,
    device const T* weight,
    device const T* mean_in,
    device const T* inv_in,
    device const T* grad_y,
    device T* grad_x,
    constant LayerNormParams& p,
    uint tid,
    uint tg_id,
    threadgroup float* tgs1,
    threadgroup float* tgs2,
    threadgroup float* shared_s1,
    threadgroup float* shared_s2
) {
    const uint row = tg_id;
    device const T* xr = x + row * p.stride_row;
    device const T* gr = grad_y + row * p.stride_row;
    device T* gx = grad_x + row * p.stride_row;

    const float mean = float(mean_in[row]);
    const float inv = float(inv_in[row]);
    const float inv_n = 1.0f / float(p.d_model);

    float s1 = 0.0f, s2 = 0.0f;
    for (uint i = tid; i < p.d_model; i += TG) {
        float v = float(xr[i]);
        float xhat = (v - mean) * inv;
        float g = float(gr[i]) * float(weight[i]);
        s1 += g;
        s2 += g * xhat;
    }
    float sg1 = simd_sum(s1); float sg2 = simd_sum(s2);
    if ((tid % SIMD) == 0) {
        const uint idx = tid / SIMD;
        tgs1[idx] = sg1;
        tgs2[idx] = sg2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t1=0, t2=0; 
        for(uint k=0; k<NSIMD; ++k) { t1+=tgs1[k]; t2+=tgs2[k]; }
        *shared_s1=t1; *shared_s2=t2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float S1 = *shared_s1, S2 = *shared_s2;

    for (uint i = tid; i < p.d_model; i += TG) {
        float v = float(xr[i]);
        float xhat = (v - mean) * inv;
        float g = float(gr[i]) * float(weight[i]);
        float dx = inv * inv_n * (float(p.d_model) * g - S1 - xhat * S2);
        gx[i] = T(dx);
    }
}

template <typename T>
inline void layernorm_bwd_x_noweight_impl(
    device const T* x,
    device const T* mean_in,
    device const T* inv_in,
    device const T* grad_y,
    device T* grad_x,
    constant LayerNormParams& p,
    uint tid,
    uint tg_id,
    threadgroup float* tgs1,
    threadgroup float* tgs2,
    threadgroup float* shared_s1,
    threadgroup float* shared_s2
) {
    const uint row = tg_id;
    device const T* xr = x + row * p.stride_row;
    device const T* gr = grad_y + row * p.stride_row;
    device T* gx = grad_x + row * p.stride_row;

    const float mean = float(mean_in[row]);
    const float inv = float(inv_in[row]);
    const float inv_n = 1.0f / float(p.d_model);

    float s1 = 0.0f, s2 = 0.0f;
    for (uint i = tid; i < p.d_model; i += TG) {
        float v = float(xr[i]);
        float xhat = (v - mean) * inv;
        float g = float(gr[i]);
        s1 += g;
        s2 += g * xhat;
    }
    float sg1 = simd_sum(s1); float sg2 = simd_sum(s2);
    if ((tid % SIMD) == 0) {
        const uint idx = tid / SIMD;
        tgs1[idx] = sg1;
        tgs2[idx] = sg2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t1=0, t2=0; 
        for(uint k=0; k<NSIMD; ++k) { t1+=tgs1[k]; t2+=tgs2[k]; }
        *shared_s1=t1; *shared_s2=t2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float S1 = *shared_s1, S2 = *shared_s2;

    for (uint i = tid; i < p.d_model; i += TG) {
        float v = float(xr[i]);
        float xhat = (v - mean) * inv;
        float g = float(gr[i]);
        float dx = inv * inv_n * (float(p.d_model) * g - S1 - xhat * S2);
        gx[i] = T(dx);
    }
}

kernel void layernorm_bwd_x_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]],
    device const half* mean_in [[ buffer(2) ]],
    device const half* inv_in [[ buffer(3) ]],
    device const half* grad_y [[ buffer(4) ]],
    device half* grad_x [[ buffer(5) ]],
    constant LayerNormParams& p [[ buffer(6) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tgs1[8], tgs2[8], shared_s1, shared_s2;
    layernorm_bwd_x_impl<half>(x, weight, mean_in, inv_in, grad_y, grad_x, p, tid, tg_id, tgs1, tgs2, &shared_s1, &shared_s2);
}

kernel void layernorm_bwd_x_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* weight [[ buffer(1) ]],
    device const float* mean_in [[ buffer(2) ]],
    device const float* inv_in [[ buffer(3) ]],
    device const float* grad_y [[ buffer(4) ]],
    device float* grad_x [[ buffer(5) ]],
    constant LayerNormParams& p [[ buffer(6) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tgs1[8], tgs2[8], shared_s1, shared_s2;
    layernorm_bwd_x_impl<float>(x, weight, mean_in, inv_in, grad_y, grad_x, p, tid, tg_id, tgs1, tgs2, &shared_s1, &shared_s2);
}

kernel void layernorm_bwd_x_noweight_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* mean_in [[ buffer(1) ]],
    device const half* inv_in [[ buffer(2) ]],
    device const half* grad_y [[ buffer(3) ]],
    device half* grad_x [[ buffer(4) ]],
    constant LayerNormParams& p [[ buffer(5) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tgs1[8], tgs2[8], shared_s1, shared_s2;
    layernorm_bwd_x_noweight_impl<half>(x, mean_in, inv_in, grad_y, grad_x, p, tid, tg_id, tgs1, tgs2, &shared_s1, &shared_s2);
}

kernel void layernorm_bwd_x_noweight_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* mean_in [[ buffer(1) ]],
    device const float* inv_in [[ buffer(2) ]],
    device const float* grad_y [[ buffer(3) ]],
    device float* grad_x [[ buffer(4) ]],
    constant LayerNormParams& p [[ buffer(5) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tgs1[8], tgs2[8], shared_s1, shared_s2;
    layernorm_bwd_x_noweight_impl<float>(x, mean_in, inv_in, grad_y, grad_x, p, tid, tg_id, tgs1, tgs2, &shared_s1, &shared_s2);
}

template <typename T>
inline void layernorm_gradw_impl(
    device const T* x,
    device const T* mean_in,
    device const T* inv_in,
    device const T* grad_y,
    device T* grad_w,
    constant LayerNormGradWParams& p,
    uint tid,
    uint tg_id,
    threadgroup float* tg_sum
) {
    const uint i = tg_id;
    if (i >= p.d_model) return;

    float acc = 0.0f;
    for (uint row = tid; row < p.rows; row += TG) {
        device const T* xr = x + row * p.stride_row;
        device const T* gr = grad_y + row * p.stride_row;
        float mean = float(mean_in[row]);
        float inv = float(inv_in[row]);
        float xhat = (float(xr[i]) - mean) * inv;
        acc += float(gr[i]) * xhat;
    }
    float sg = simd_sum(acc);
    if ((tid % SIMD) == 0) tg_sum[tid/SIMD] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot=0; for(uint k=0; k<NSIMD; ++k) tot+=tg_sum[k];
        grad_w[i] = T(tot);
    }
}

template <typename T>
inline void layernorm_gradb_impl(
    device const T* grad_y,
    device T* grad_b,
    constant LayerNormGradWParams& p,
    uint tid,
    uint tg_id,
    threadgroup float* tg_sum
) {
    const uint i = tg_id;
    if (i >= p.d_model) return;

    float acc = 0.0f;
    for (uint row = tid; row < p.rows; row += TG) {
        device const T* gr = grad_y + row * p.stride_row;
        acc += float(gr[i]);
    }
    float sg = simd_sum(acc);
    if ((tid % SIMD) == 0) tg_sum[tid/SIMD] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot=0; for(uint k=0; k<NSIMD; ++k) tot+=tg_sum[k];
        grad_b[i] = T(tot);
    }
}

kernel void layernorm_gradw_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* mean_in [[ buffer(1) ]],
    device const half* inv_in [[ buffer(2) ]],
    device const half* grad_y [[ buffer(3) ]],
    device half* grad_w [[ buffer(4) ]],
    constant LayerNormGradWParams& p [[ buffer(5) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{ 
    threadgroup float tg_sum[8];
    layernorm_gradw_impl<half>(x, mean_in, inv_in, grad_y, grad_w, p, tid, tg_id, tg_sum); 
}

kernel void layernorm_gradw_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* mean_in [[ buffer(1) ]],
    device const float* inv_in [[ buffer(2) ]],
    device const float* grad_y [[ buffer(3) ]],
    device float* grad_w [[ buffer(4) ]],
    constant LayerNormGradWParams& p [[ buffer(5) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{ 
    threadgroup float tg_sum[8];
    layernorm_gradw_impl<float>(x, mean_in, inv_in, grad_y, grad_w, p, tid, tg_id, tg_sum); 
}

kernel void layernorm_gradb_fp16(
    device const half* grad_y [[ buffer(0) ]],
    device half* grad_b [[ buffer(1) ]],
    constant LayerNormGradWParams& p [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{ 
    threadgroup float tg_sum[8];
    layernorm_gradb_impl<half>(grad_y, grad_b, p, tid, tg_id, tg_sum); 
}

kernel void layernorm_gradb_fp32(
    device const float* grad_y [[ buffer(0) ]],
    device float* grad_b [[ buffer(1) ]],
    constant LayerNormGradWParams& p [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{ 
    threadgroup float tg_sum[8];
    layernorm_gradb_impl<float>(grad_y, grad_b, p, tid, tg_id, tg_sum); 
}
