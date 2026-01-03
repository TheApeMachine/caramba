#include <metal_stdlib>
using namespace metal;

// Must match `RMSNormParams` in `ops.mm` (layout + types).
struct RMSNormParams {
    uint d_model;
    float eps;
    uint stride_row; // in elements
};

struct RMSNormGradWParams {
    uint d_model;
    uint rows;
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

kernel void rmsnorm_fwd_inv_fp16(
    device const half* x      [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]],
    device half* out          [[ buffer(2) ]],
    device half* inv_out      [[ buffer(3) ]], // (rows,)
    constant RMSNormParams& p [[ buffer(4) ]],
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
        const float inv = rsqrt(mean + p.eps);
        shared_inv = inv;
        inv_out[row] = half(inv);
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

kernel void rmsnorm_noweight_fwd_inv_fp16(
    device const half* x      [[ buffer(0) ]],
    device half* out          [[ buffer(1) ]],
    device half* inv_out      [[ buffer(2) ]], // (rows,)
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
        const float inv = rsqrt(mean + p.eps);
        shared_inv = inv;
        inv_out[row] = half(inv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv = shared_inv;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        yr[i] = half(v * inv);
    }
}

kernel void rmsnorm_bwd_fp16(
    device const half* x      [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]],
    device const half* inv_in [[ buffer(2) ]], // (rows,)
    device const half* grad_y [[ buffer(3) ]],
    device half* grad_x       [[ buffer(4) ]],
    constant RMSNormParams& p [[ buffer(5) ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint tg_id                [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD; // 8

    const uint row = tg_id;
    device const half* xr = x + row * p.stride_row;
    device const half* gr = grad_y + row * p.stride_row;
    device half* gx = grad_x + row * p.stride_row;

    threadgroup float tg_sum[NSIMD];
    threadgroup float shared_s;

    const float inv = float(inv_in[row]);

    float sum = 0.0f;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        const float w = float(weight[i]);
        const float gy = float(gr[i]);
        sum += gy * v * w;
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
        shared_s = total / float(p.d_model);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float s = shared_s;
    const float inv3 = inv * inv * inv;

    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        const float w = float(weight[i]);
        const float gy = float(gr[i]);
        const float dx = gy * inv * w - v * inv3 * s;
        gx[i] = half(dx);
    }
}

kernel void rmsnorm_bwd_noweight_fp16(
    device const half* x      [[ buffer(0) ]],
    device const half* inv_in [[ buffer(1) ]], // (rows,)
    device const half* grad_y [[ buffer(2) ]],
    device half* grad_x       [[ buffer(3) ]],
    constant RMSNormParams& p [[ buffer(4) ]],
    uint tid                  [[ thread_position_in_threadgroup ]],
    uint tg_id                [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;
    constexpr uint SIMD = 32;
    constexpr uint NSIMD = TG / SIMD; // 8

    const uint row = tg_id;
    device const half* xr = x + row * p.stride_row;
    device const half* gr = grad_y + row * p.stride_row;
    device half* gx = grad_x + row * p.stride_row;

    threadgroup float tg_sum[NSIMD];
    threadgroup float shared_s;

    const float inv = float(inv_in[row]);

    float sum = 0.0f;
    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        const float gy = float(gr[i]);
        sum += gy * v;
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
        shared_s = total / float(p.d_model);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float s = shared_s;
    const float inv3 = inv * inv * inv;

    for (uint i = tid; i < p.d_model; i += TG) {
        const float v = float(xr[i]);
        const float gy = float(gr[i]);
        const float dx = gy * inv - v * inv3 * s;
        gx[i] = half(dx);
    }
}

kernel void rmsnorm_gradw_fp16(
    device const half* x           [[ buffer(0) ]],
    device const half* inv_in      [[ buffer(1) ]], // (rows,)
    device const half* grad_y      [[ buffer(2) ]],
    device half* grad_w            [[ buffer(3) ]], // (d_model,)
    constant RMSNormGradWParams& p [[ buffer(4) ]],
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
        grad_w[i] = half(total);
    }
}

