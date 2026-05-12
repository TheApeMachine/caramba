#include <metal_stdlib>
using namespace metal;

kernel void ai_free_energy_terms_kernel(
    device const float* mu           [[buffer(0)]],
    device const float* log_sigma    [[buffer(1)]],
    device       float* terms        [[buffer(2)]],
    constant     uint&  n            [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float m  = mu[gid];
    float ls = log_sigma[gid];
    float el = exp(ls);
    terms[gid] = 0.5f * (m * m + el - ls - 1.f);
}

kernel void ai_reduce_fe_atomic_kernel(
    device const float* terms [[buffer(0)]],
    device atomic_float* sum [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    atomic_fetch_add_explicit(sum, terms[gid], memory_order_relaxed);
}

kernel void ai_finalize_free_energy_kernel(
    device atomic_float* sum [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    float s = atomic_load_explicit(sum, memory_order_relaxed);
    out[0] = s;
}

kernel void ai_belief_update_kernel(
    device const float* mu           [[buffer(0)]],
    device const float* log_sigma    [[buffer(1)]],
    device const float* pred_err     [[buffer(2)]],
    device       float* out            [[buffer(3)]],
    constant     float& lr           [[buffer(4)]],
    constant     uint&  n            [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float m  = mu[gid];
    float ls = log_sigma[gid];
    float pe = pred_err[gid];
    out[gid]     = m - lr * (m + pe);
    float el     = exp(ls);
    out[n + gid] = ls - lr * (el - 1.f);
}

#define LOG_PREC_MIN (-80.f)
#define LOG_PREC_MAX (80.f)

kernel void ai_precision_weight_kernel(
    device const float* err      [[buffer(0)]],
    device const float* log_prec [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    constant     uint&  n        [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float lp = log_prec[gid];
    lp = clamp(lp, LOG_PREC_MIN, LOG_PREC_MAX);
    float e = exp(lp);
    out[gid] = err[gid] * e;
}

kernel void ai_expected_free_energy_kernel(
    device const float* q [[buffer(0)]],
    device       float* out [[buffer(1)]],
    constant     uint&  n   [[buffer(2)]],
    constant     uint&  K   [[buffer(3)]],
    constant     float& eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= K) {
        return;
    }
    uint k = gid;
    float acc = 0.f;
    for (uint i = 0; i < n; i++) {
        float qq = q[i * K + k];
        acc -= qq * log(qq + eps);
    }
    out[k] = acc;
}
