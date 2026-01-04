#include <metal_stdlib>
using namespace metal;

// Parameters for SSM selective scan.
struct SSMScanParams {
    uint B;
    uint T;
    uint D_inner;
    uint D_state;
};

// Layout assumptions (validated in ops.mm):
// - x, dt: (B,T,D_inner) contiguous
// - A: (D_inner,D_state) contiguous
// - Bv, Cv: (B,T,D_state) contiguous
// - D: (D_inner,) contiguous
// - y: (B,T,D_inner) contiguous
// - h_hist, g_hist: (B,T,D_inner,D_state) contiguous (state fastest)

kernel void ssm_scan_fwd_fp16(
    device const half* x         [[ buffer(0) ]],
    device const half* dt        [[ buffer(1) ]],
    device const half* A         [[ buffer(2) ]],
    device const half* Bv        [[ buffer(3) ]],
    device const half* Cv        [[ buffer(4) ]],
    device const half* D         [[ buffer(5) ]],
    device half* y               [[ buffer(6) ]],
    device half* h_hist          [[ buffer(7) ]],
    constant SSMScanParams& p    [[ buffer(8) ]],
    uint tid                     [[ thread_index_in_threadgroup ]],
    uint tg_id                   [[ threadgroup_position_in_grid ]]
) {
    // Threadgroup is expected to be 32 threads; we use only lanes < D_state.
    const uint lane = tid;
    if (tg_id >= (p.B * p.D_inner)) {
        return;
    }
    const uint b = tg_id / p.D_inner;
    const uint d = tg_id - b * p.D_inner;
    if (b >= p.B || d >= p.D_inner) {
        return;
    }
    if (p.D_state == 0 || p.D_state > 32) {
        return;
    }

    const bool active = lane < p.D_state;
    const uint a_off = (d * p.D_state + lane);
    const float a_const = active ? float(A[a_off]) : 0.0f;
    float h = 0.0f;

    // Preload skip D[d]
    const float d_skip = float(D[d]);

    for (uint t = 0; t < p.T; ++t) {
        const uint x_off = (b * p.T + t) * p.D_inner + d;
        const float x_t = float(x[x_off]);
        const float dt_t = float(dt[x_off]);

        float g = 0.0f;
        if (active) {
            const float a_t = exp(dt_t * a_const);
            const uint bt_off = (b * p.T + t) * p.D_state + lane;
            const float b_t = float(Bv[bt_off]);
            const float u_t = dt_t * b_t * x_t;
            h = a_t * h + u_t;
            // Persist h history for backward.
            const uint h_off = ((b * p.T + t) * p.D_inner + d) * p.D_state + lane;
            h_hist[h_off] = half(h);
            const float c_t = float(Cv[bt_off]);
            g = h * c_t;
        }

        // Reduce sum over lanes (max 32).
        const float sum = simd_sum(g);
        if (lane == 0) {
            y[x_off] = half(sum + d_skip * x_t);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void ssm_scan_bwd_g_fp16(
    device const half* x         [[ buffer(0) ]],
    device const half* dt        [[ buffer(1) ]],
    device const half* A         [[ buffer(2) ]],
    device const half* Bv        [[ buffer(3) ]],
    device const half* Cv        [[ buffer(4) ]],
    device const half* D         [[ buffer(5) ]],
    device const half* grad_y    [[ buffer(6) ]],
    device const half* h_hist    [[ buffer(7) ]],
    device half* g_hist          [[ buffer(8) ]],
    device half* grad_x          [[ buffer(9) ]],
    device half* grad_dt         [[ buffer(10) ]],
    device half* gradA_partial   [[ buffer(11) ]], // (B,D_inner,D_state)
    constant SSMScanParams& p    [[ buffer(12) ]],
    uint tid                     [[ thread_index_in_threadgroup ]],
    uint tg_id                   [[ threadgroup_position_in_grid ]]
) {
    const uint lane = tid;
    if (tg_id >= (p.B * p.D_inner)) {
        return;
    }
    const uint b = tg_id / p.D_inner;
    const uint d = tg_id - b * p.D_inner;
    if (b >= p.B || d >= p.D_inner) {
        return;
    }
    if (p.D_state == 0 || p.D_state > 32) {
        return;
    }

    const bool active = lane < p.D_state;
    const uint a_off = (d * p.D_state + lane);
    const float a_const = active ? float(A[a_off]) : 0.0f;
    const float d_skip = float(D[d]);

    float g_next = 0.0f;
    float accum_A = 0.0f;

    for (int tt = int(p.T) - 1; tt >= 0; --tt) {
        const uint t = uint(tt);
        const uint x_off = (b * p.T + t) * p.D_inner + d;
        const float x_t = float(x[x_off]);
        const float dt_t = float(dt[x_off]);
        const float gy_t = float(grad_y[x_off]);

        float g = 0.0f;
        float a_next = 0.0f;
        if (active && (t + 1) < p.T) {
            const uint x_off_n = (b * p.T + (t + 1)) * p.D_inner + d;
            const float dt_n = float(dt[x_off_n]);
            a_next = exp(dt_n * a_const);
        }
        if (active) {
            const uint bt_off = (b * p.T + t) * p.D_state + lane;
            const float c_t = float(Cv[bt_off]);
            g = gy_t * c_t + a_next * g_next;

            const uint g_off = ((b * p.T + t) * p.D_inner + d) * p.D_state + lane;
            g_hist[g_off] = half(g);

            // gradA accumulation uses h_{t-1}.
            float h_prev = 0.0f;
            if (t > 0) {
                const uint h_off_prev = ((b * p.T + (t - 1)) * p.D_inner + d) * p.D_state + lane;
                h_prev = float(h_hist[h_off_prev]);
            }
            const float a_t = exp(dt_t * a_const);
            const float grad_a = g * h_prev;
            accum_A += grad_a * a_t * dt_t;
        }

        // grad_x and grad_dt are reduced over state lanes.
        float gx_u = 0.0f;
        float gdt_u = 0.0f;
        float gdt_a = 0.0f;
        if (active) {
            const uint bt_off = (b * p.T + t) * p.D_state + lane;
            const float b_t = float(Bv[bt_off]);
            gx_u = g * dt_t * b_t;
            gdt_u = g * b_t * x_t;

            float h_prev = 0.0f;
            if (t > 0) {
                const uint h_off_prev = ((b * p.T + (t - 1)) * p.D_inner + d) * p.D_state + lane;
                h_prev = float(h_hist[h_off_prev]);
            }
            const float a_t = exp(dt_t * a_const);
            const float grad_a = g * h_prev;
            gdt_a = grad_a * a_t * a_const;
        }
        const float gx_sum = simd_sum(gx_u);
        const float gdt_sum = simd_sum(gdt_u + gdt_a);
        if (lane == 0) {
            grad_x[x_off] = half(gy_t * d_skip + gx_sum);
            grad_dt[x_off] = half(gdt_sum);
        }

        if (active) {
            g_next = g;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        const uint ao = (b * p.D_inner + d) * p.D_state + lane;
        gradA_partial[ao] = half(accum_A);
    }
}

kernel void ssm_gradB_reduce_fp16(
    device const half* x         [[ buffer(0) ]],
    device const half* dt        [[ buffer(1) ]],
    device const half* g_hist    [[ buffer(2) ]],
    device half* grad_B          [[ buffer(3) ]], // (B,T,D_state)
    constant SSMScanParams& p    [[ buffer(4) ]],
    uint tid                     [[ thread_index_in_threadgroup ]],
    uint tg_id                   [[ threadgroup_position_in_grid ]]
) {
    // One threadgroup per (b,t,s), 256 threads reducing over d.
    const uint s = tg_id % p.D_state;
    const uint bt = tg_id / p.D_state;
    const uint b = bt / p.T;
    const uint t = bt - b * p.T;
    if (b >= p.B || t >= p.T || s >= p.D_state) {
        return;
    }

    float acc = 0.0f;
    for (uint d = tid; d < p.D_inner; d += 256) {
        const uint x_off = (b * p.T + t) * p.D_inner + d;
        const float x_t = float(x[x_off]);
        const float dt_t = float(dt[x_off]);
        const uint g_off = ((b * p.T + t) * p.D_inner + d) * p.D_state + s;
        const float g = float(g_hist[g_off]);
        acc += g * dt_t * x_t;
    }
    const float sum = simd_sum(acc);
    if ((tid % 32) == 0) {
        // First lane of each simdgroup writes to threadgroup scratch via atomic-free reduction.
        // We rely on the small D_inner (512) and fixed TG for correctness/perf.
    }
    threadgroup float tg_sum[8];
    if ((tid % 32) == 0) {
        tg_sum[tid / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total += tg_sum[i];
        }
        grad_B[(b * p.T + t) * p.D_state + s] = half(total);
    }
}

kernel void ssm_gradC_reduce_fp16(
    device const half* grad_y    [[ buffer(0) ]],
    device const half* h_hist    [[ buffer(1) ]],
    device half* grad_C          [[ buffer(2) ]], // (B,T,D_state)
    constant SSMScanParams& p    [[ buffer(3) ]],
    uint tid                     [[ thread_index_in_threadgroup ]],
    uint tg_id                   [[ threadgroup_position_in_grid ]]
) {
    const uint s = tg_id % p.D_state;
    const uint bt = tg_id / p.D_state;
    const uint b = bt / p.T;
    const uint t = bt - b * p.T;
    if (b >= p.B || t >= p.T || s >= p.D_state) {
        return;
    }

    float acc = 0.0f;
    for (uint d = tid; d < p.D_inner; d += 256) {
        const uint y_off = (b * p.T + t) * p.D_inner + d;
        const float gy = float(grad_y[y_off]);
        const uint h_off = ((b * p.T + t) * p.D_inner + d) * p.D_state + s;
        const float h = float(h_hist[h_off]);
        acc += gy * h;
    }
    const float sum = simd_sum(acc);
    threadgroup float tg_sum[8];
    if ((tid % 32) == 0) {
        tg_sum[tid / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total += tg_sum[i];
        }
        grad_C[(b * p.T + t) * p.D_state + s] = half(total);
    }
}

kernel void ssm_gradD_reduce_fp16(
    device const half* x         [[ buffer(0) ]],
    device const half* grad_y    [[ buffer(1) ]],
    device half* grad_D          [[ buffer(2) ]], // (D_inner,)
    constant SSMScanParams& p    [[ buffer(3) ]],
    uint tid                     [[ thread_index_in_threadgroup ]],
    uint tg_id                   [[ threadgroup_position_in_grid ]]
) {
    // One threadgroup per d, reduce over (b,t).
    const uint d = tg_id;
    if (d >= p.D_inner) {
        return;
    }
    const uint N = p.B * p.T;
    float acc = 0.0f;
    for (uint i = tid; i < N; i += 256) {
        const uint b = i / p.T;
        const uint t = i - b * p.T;
        const uint off = (b * p.T + t) * p.D_inner + d;
        acc += float(grad_y[off]) * float(x[off]);
    }
    const float sum = simd_sum(acc);
    threadgroup float tg_sum[8];
    if ((tid % 32) == 0) {
        tg_sum[tid / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total += tg_sum[i];
        }
        grad_D[d] = half(total);
    }
}

kernel void ssm_gradA_reduce_fp16(
    device const half* gradA_partial [[ buffer(0) ]], // (B,D_inner,D_state)
    device half* grad_A              [[ buffer(1) ]], // (D_inner,D_state)
    constant SSMScanParams& p        [[ buffer(2) ]],
    uint tid                         [[ thread_index_in_threadgroup ]],
    uint tg_id                       [[ threadgroup_position_in_grid ]]
) {
    // One threadgroup per (d,s), reduce over batch B.
    const uint s = tg_id % p.D_state;
    const uint d = tg_id / p.D_state;
    if (d >= p.D_inner || s >= p.D_state) {
        return;
    }
    float acc = 0.0f;
    for (uint b = tid; b < p.B; b += 256) {
        const uint off = (b * p.D_inner + d) * p.D_state + s;
        acc += float(gradA_partial[off]);
    }
    const float sum = simd_sum(acc);
    threadgroup float tg_sum[8];
    if ((tid % 32) == 0) {
        tg_sum[tid / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total += tg_sum[i];
        }
        grad_A[d * p.D_state + s] = half(total);
    }
}

