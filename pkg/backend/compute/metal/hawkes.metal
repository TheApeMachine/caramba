#include <metal_stdlib>
using namespace metal;

kernel void hawkes_intensity_kernel(
    device const float* times [[buffer(0)]],
    device const float* alpha [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device const float* mu [[buffer(3)]],
    device       float* out [[buffer(4)]],
    constant     float& t [[buffer(5)]],
    constant     uint&  K [[buffer(6)]],
    constant     uint&  T [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= K) {
        return;
    }
    float bk = beta[gid];
    float acc = 0.f;
    for (uint idx = 0; idx < T; idx++) {
        float dt = t - times[idx];
        if (dt <= 0.f) {
            break;
        }
        acc += exp(-bk * dt);
    }
    out[gid] = mu[gid] + alpha[gid] * acc;
}

kernel void hawkes_kernel_matrix_kernel(
    device const float* times [[buffer(0)]],
    device       float* out [[buffer(1)]],
    constant     uint&  T [[buffer(2)]],
    constant     float& alpha [[buffer(3)]],
    constant     float& beta [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.x;
    uint col = gid.y;
    if (row >= T || col >= T) {
        return;
    }
    if (col > row) {
        float dt = times[col] - times[row];
        out[row * T + col] = alpha * exp(-beta * dt);
    } else {
        out[row * T + col] = 0.f;
    }
}

kernel void hawkes_log_term_kernel(
    device const float* intensities [[buffer(0)]],
    device       float* partials [[buffer(1)]],
    constant     uint&  T [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= T) {
        return;
    }
    float lam = intensities[gid];
    partials[gid] = (lam > 0.f) ? log(lam) : 0.f;
}

kernel void hawkes_reduce_sum_atomic_kernel(
    device const float* src [[buffer(0)]],
    device atomic_float* sum [[buffer(1)]],
    constant     uint&  n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    atomic_fetch_add_explicit(sum, src[gid], memory_order_relaxed);
}

kernel void hawkes_loglik_finalize_kernel(
    device atomic_float* sum [[buffer(0)]],
    constant     float& integral [[buffer(1)]],
    device       float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    float s = atomic_load_explicit(sum, memory_order_relaxed);
    out[0] = s - integral;
}

kernel void hawkes_sim_clear_kernel(
    device float* out [[buffer(0)]],
    constant     uint& total [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total) {
        return;
    }
    out[gid] = -1.f;
}

// One thread per dimension — Ogata thinning on-device; float uniforms (Metal has no double in kernels).
kernel void hawkes_simulate_dim_kernel(
    device const float* mu [[buffer(0)]],
    device const float* alpha [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device       float* out [[buffer(3)]],
    constant     float& T_max [[buffer(4)]],
    constant     uint&  K [[buffer(5)]],
    constant     uint&  maxSteps [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= K) {
        return;
    }
    uint k = gid;
    ulong seed = (ulong)(k + 1u) * 6364136223846793005ul + 1442695040888963407ul;
    device float* kevents = out + (ulong)k * (ulong)maxSteps;
    float tEvent = 0.f;
    uint count = 0u;
    float muk = mu[k];
    float alphak = alpha[k];
    float betak = beta[k];

    while (tEvent < T_max && count < maxSteps) {
        float lstar = muk;
        for (uint i = 0u; i < count; i++) {
            lstar += alphak * exp(-betak * (tEvent - kevents[i]));
        }
        seed = seed * 6364136223846793005ul + 1442695040888963407ul;
        ulong t1 = seed >> 11;
        // (0,1) uniform from 24 high bits; Metal has no double in kernel code.
        float u1 = (float)(t1 & 0xFFFFFFul) * (1.f / 16777216.f);
        u1 = fmax(u1, 1e-20f);
        float dt = -log(u1) / lstar;
        tEvent += dt;
        if (tEvent >= T_max) {
            break;
        }
        float lam = muk;
        for (uint j = 0u; j < count; j++) {
            lam += alphak * exp(-betak * (tEvent - kevents[j]));
        }
        seed = seed * 6364136223846793005ul + 1442695040888963407ul;
        ulong t2 = seed >> 11;
        float u2 = (float)(t2 & 0xFFFFFFul) * (1.f / 16777216.f);
        if (u2 <= lam / lstar) {
            kevents[count++] = tEvent;
        }
    }
}
