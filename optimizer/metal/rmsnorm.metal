#include <metal_stdlib>
using namespace metal;

// Must match `RMSNormParams` in `ops.mm` (layout + types).
struct RMSNormParams {
    uint d_model;
    float eps;
    uint stride_row; // in elements
};

// Must match `RMSNormGradWParams` in `ops.mm`.
struct RMSNormGradWParams {
    uint d_model;
    uint rows;
    uint stride_row; // in elements
};

// Constants shared across kernels
constant uint TG = 256;
constant uint SIMD = 32;
constant uint NSIMD = TG / SIMD; // 8

template <typename T, bool HAS_WEIGHT, bool HAS_BWD_INV>
inline void rmsnorm_fwd_impl(
    device const T* x,
    device const T* weight,
    device T* out,
    device T* inv_out, // Optional, depending on HAS_BWD_INV
    constant RMSNormParams& p,
    uint tid,
    uint tg_id,
    threadgroup float* tg_sum,
    threadgroup float* shared_inv
) {
    const uint row = tg_id;
    device const T* xr = x + row * p.stride_row;
    device T* yr = out + row * p.stride_row;

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
        const float inv = rsqrt(mean + p.eps);
        *shared_inv = inv;
        if constexpr (HAS_BWD_INV) {
            inv_out[row] = T(inv);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv = *shared_inv;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        float val = v * inv;
        if constexpr (HAS_WEIGHT) {
            val *= float(weight[i]);
        }
        yr[i] = T(val);
    }
}

// ------ FP16 Instantiations ------
kernel void rmsnorm_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]],
    device half* out [[ buffer(2) ]],
    constant RMSNormParams& p [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<half, true, false>(x, weight, out, nullptr, p, tid, tg_id, tg_sum, &shared_inv);
}

kernel void rmsnorm_fwd_inv_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]],
    device half* out [[ buffer(2) ]],
    device half* inv_out [[ buffer(3) ]],
    constant RMSNormParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<half, true, true>(x, weight, out, inv_out, p, tid, tg_id, tg_sum, &shared_inv);
}

kernel void rmsnorm_noweight_fp16(
    device const half* x [[ buffer(0) ]],
    device half* out [[ buffer(1) ]],
    constant RMSNormParams& p [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<half, false, false>(x, nullptr, out, nullptr, p, tid, tg_id, tg_sum, &shared_inv);
}

kernel void rmsnorm_noweight_fwd_inv_fp16(
    device const half* x [[ buffer(0) ]],
    device half* out [[ buffer(1) ]],
    device half* inv_out [[ buffer(2) ]],
    constant RMSNormParams& p [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<half, false, true>(x, nullptr, out, inv_out, p, tid, tg_id, tg_sum, &shared_inv);
}

// ------ FP32 Instantiations ------
kernel void rmsnorm_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* weight [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    constant RMSNormParams& p [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<float, true, false>(x, weight, out, nullptr, p, tid, tg_id, tg_sum, &shared_inv);
}

kernel void rmsnorm_fwd_inv_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* weight [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    device float* inv_out [[ buffer(3) ]],
    constant RMSNormParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<float, true, true>(x, weight, out, inv_out, p, tid, tg_id, tg_sum, &shared_inv);
}

kernel void rmsnorm_noweight_fp32(
    device const float* x [[ buffer(0) ]],
    device float* out [[ buffer(1) ]],
    constant RMSNormParams& p [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<float, false, false>(x, nullptr, out, nullptr, p, tid, tg_id, tg_sum, &shared_inv);
}

kernel void rmsnorm_noweight_fwd_inv_fp32(
    device const float* x [[ buffer(0) ]],
    device float* out [[ buffer(1) ]],
    device float* inv_out [[ buffer(2) ]],
    constant RMSNormParams& p [[ buffer(3) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tg_sum[8];
    threadgroup float shared_inv;
    rmsnorm_fwd_impl<float, false, true>(x, nullptr, out, inv_out, p, tid, tg_id, tg_sum, &shared_inv);
}


// ------ Backward Kernels ------

template <typename T, bool HAS_WEIGHT>
inline void rmsnorm_bwd_impl(
    device const T* x,
    device const T* weight, // Optional
    device const T* inv_in,
    device const T* grad_y,
    device T* grad_x,
    constant RMSNormParams& p,
    uint tid,
    uint tg_id,
    threadgroup float* tg_sum,
    threadgroup float* shared_s
) {
    const uint row = tg_id;
    device const T* xr = x + row * p.stride_row;
    device const T* gr = grad_y + row * p.stride_row;
    device T* gx = grad_x + row * p.stride_row;

    const float inv = float(inv_in[row]);

    float sum = 0.0f;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        const float gy = float(gr[i]);
        float val = gy * v;
        if constexpr (HAS_WEIGHT) {
            val *= float(weight[i]);
        }
        sum += val;
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
        *shared_s = total / float(p.d_model);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float s = *shared_s;
    const float inv3 = inv * inv * inv;

    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        const float gy = float(gr[i]);
        float term1 = gy;
        if constexpr (HAS_WEIGHT) {
            term1 *= float(weight[i]);
        }
        const float dx = term1 * inv - v * inv3 * s;
        gx[i] = T(dx);
    }
}

kernel void rmsnorm_bwd_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]],
    device const half* inv_in [[ buffer(2) ]],
    device const half* grad_y [[ buffer(3) ]],
    device half* grad_x [[ buffer(4) ]],
    constant RMSNormParams& p [[ buffer(5) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tg_sum[8];
    threadgroup float shared_s;
    rmsnorm_bwd_impl<half, true>(x, weight, inv_in, grad_y, grad_x, p, tid, tg_id, tg_sum, &shared_s);
}

kernel void rmsnorm_bwd_noweight_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* inv_in [[ buffer(1) ]],
    device const half* grad_y [[ buffer(2) ]],
    device half* grad_x [[ buffer(3) ]],
    constant RMSNormParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tg_sum[8];
    threadgroup float shared_s;
    rmsnorm_bwd_impl<half, false>(x, nullptr, inv_in, grad_y, grad_x, p, tid, tg_id, tg_sum, &shared_s);
}

kernel void rmsnorm_bwd_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* weight [[ buffer(1) ]],
    device const float* inv_in [[ buffer(2) ]],
    device const float* grad_y [[ buffer(3) ]],
    device float* grad_x [[ buffer(4) ]],
    constant RMSNormParams& p [[ buffer(5) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tg_sum[8];
    threadgroup float shared_s;
    rmsnorm_bwd_impl<float, true>(x, weight, inv_in, grad_y, grad_x, p, tid, tg_id, tg_sum, &shared_s);
}

kernel void rmsnorm_bwd_noweight_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* inv_in [[ buffer(1) ]],
    device const float* grad_y [[ buffer(2) ]],
    device float* grad_x [[ buffer(3) ]],
    constant RMSNormParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tg_sum[8];
    threadgroup float shared_s;
    rmsnorm_bwd_impl<float, false>(x, nullptr, inv_in, grad_y, grad_x, p, tid, tg_id, tg_sum, &shared_s);
}

// ------ Grad Weight Kernels ------

template <typename T>
inline void rmsnorm_gradw_impl(
    device const T* x,
    device const T* inv_in,
    device const T* grad_y,
    device T* grad_w,
    constant RMSNormGradWParams& p,
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
        const float inv = float(inv_in[row]);
        acc += float(gr[i]) * float(xr[i]) * inv;
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
        grad_w[i] = T(total);
    }
}

kernel void rmsnorm_gradw_fp16(
    device const half* x [[ buffer(0) ]],
    device const half* inv_in [[ buffer(1) ]],
    device const half* grad_y [[ buffer(2) ]],
    device half* grad_w [[ buffer(3) ]],
    constant RMSNormGradWParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tg_sum[8];
    rmsnorm_gradw_impl<half>(x, inv_in, grad_y, grad_w, p, tid, tg_id, tg_sum);
}

kernel void rmsnorm_gradw_fp32(
    device const float* x [[ buffer(0) ]],
    device const float* inv_in [[ buffer(1) ]],
    device const float* grad_y [[ buffer(2) ]],
    device float* grad_w [[ buffer(3) ]],
    constant RMSNormGradWParams& p [[ buffer(4) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_id [[ threadgroup_position_in_grid ]]) 
{
    threadgroup float tg_sum[8];
    rmsnorm_gradw_impl<float>(x, inv_in, grad_y, grad_w, p, tid, tg_id, tg_sum);
}
