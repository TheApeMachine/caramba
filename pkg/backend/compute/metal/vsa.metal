#include <metal_stdlib>
using namespace metal;

// VSA binding: elementwise multiply
kernel void vsa_bind_kernel(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device       float* out    [[buffer(2)]],
    constant     uint&  n      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        out[gid] = a[gid] * b[gid];
    }
}

kernel void vsa_square_kernel(
    device const float* in  [[buffer(0)]],
    device       float* out [[buffer(1)]],
    constant     uint&  n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        float v = in[gid];
        out[gid] = v * v;
    }
}

kernel void vsa_reduce_sum_atomic_kernel(
    device const float* src [[buffer(0)]],
    device atomic_float* sum [[buffer(1)]],
    constant     uint&  n    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    atomic_fetch_add_explicit(sum, src[gid], memory_order_relaxed);
}

kernel void vsa_finalize_dot_kernel(
    device atomic_float* sum [[buffer(0)]],
    device float* out        [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    float s = atomic_load_explicit(sum, memory_order_relaxed);
    out[0] = s;
}

kernel void vsa_finalize_invnorm_kernel(
    device atomic_float* sumsq [[buffer(0)]],
    device float* inv_norm     [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    float s = atomic_load_explicit(sumsq, memory_order_relaxed);
    float inv = (s > 1e-24f) ? (1.f / sqrt(s)) : 1.f;
    inv_norm[0] = inv;
}

// VSA dot product: elementwise multiply — device reduction into out[0]
kernel void vsa_mul_kernel(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device       float* out    [[buffer(2)]],
    constant     uint&  n      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        out[gid] = a[gid] * b[gid];
    }
}

// VSA L2-normalise: out[i] = in[i] * inv_norm
kernel void vsa_scale_kernel(
    device const float* in       [[buffer(0)]],
    device       float* out      [[buffer(1)]],
    constant     float& inv_norm [[buffer(2)]],
    constant     uint&  n        [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        out[gid] = in[gid] * inv_norm;
    }
}

kernel void vsa_bundle_sum_kernel(
    device const float* vectors [[buffer(0)]],
    device       float* out     [[buffer(1)]],
    constant     uint&  count   [[buffer(2)]],
    constant     uint&  n       [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }

    float sum = 0.0f;
    for (uint vector = 0; vector < count; vector++) {
        sum += vectors[vector * n + gid];
    }
    out[gid] = sum;
}

kernel void vsa_permute_kernel(
    device const float* src   [[buffer(0)]],
    device       float* out   [[buffer(1)]],
    constant     int&   shift [[buffer(2)]],
    constant     uint&  n     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }

    int normalized = shift % int(n);
    if (normalized < 0) {
        normalized += int(n);
    }

    int source = int(gid) - normalized;
    if (source < 0) {
        source += int(n);
    }

    out[gid] = src[source];
}
