#include <metal_stdlib>
using namespace metal;

// VSA binding: elementwise multiply
kernel void vsa_bind_kernel(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device       float* out    [[buffer(2)]],
    constant     int&   n      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid < n) out[gid] = a[gid] * b[gid];
}

// VSA dot product: elementwise multiply — result requires host-side sum reduction
kernel void vsa_mul_kernel(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device       float* out    [[buffer(2)]],
    constant     int&   n      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid < n) out[gid] = a[gid] * b[gid];
}

// VSA L2-normalise: out[i] = in[i] * inv_norm
kernel void vsa_scale_kernel(
    device const float* in       [[buffer(0)]],
    device       float* out      [[buffer(1)]],
    constant     float& inv_norm [[buffer(2)]],
    constant     int&   n        [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid < n) out[gid] = in[gid] * inv_norm;
}
