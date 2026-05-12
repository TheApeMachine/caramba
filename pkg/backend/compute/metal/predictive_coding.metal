#include <metal_stdlib>
using namespace metal;

kernel void pc_prediction_kernel(
    device const float* W [[buffer(0)]],
    device const float* r [[buffer(1)]],
    device       float* dst [[buffer(2)]],
    constant     uint&  D_out [[buffer(3)]],
    constant     uint&  D_in [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= D_out) {
        return;
    }
    float s = 0.f;
    for (uint j = 0; j < D_in; j++) {
        s += W[gid * D_in + j] * r[j];
    }
    dst[gid] = s;
}

kernel void pc_prediction_error_kernel(
    device const float* x [[buffer(0)]],
    device const float* mu_hat [[buffer(1)]],
    device const float* prec [[buffer(2)]],
    device       float* dst [[buffer(3)]],
    constant     uint&  n [[buffer(4)]],
    constant     uint&  use_prec [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float e = x[gid] - mu_hat[gid];
    if (use_prec != 0) {
        e *= prec[gid];
    }
    dst[gid] = e;
}

kernel void pc_update_representation_kernel(
    device const float* r [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* eps_lower [[buffer(2)]],
    device const float* eps_self [[buffer(3)]],
    device       float* dst [[buffer(4)]],
    constant     float& lr [[buffer(5)]],
    constant     uint&  D_out [[buffer(6)]],
    constant     uint&  D_in [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= D_in) {
        return;
    }
    uint j = gid;
    float acc = -eps_self[j];
    for (uint i = 0; i < D_out; i++) {
        acc += W[i * D_in + j] * eps_lower[i];
    }
    dst[j] = r[j] + lr * acc;
}

kernel void pc_update_weights_kernel(
    device const float* W [[buffer(0)]],
    device const float* eps [[buffer(1)]],
    device const float* r [[buffer(2)]],
    device       float* dst [[buffer(3)]],
    constant     float& lr [[buffer(4)]],
    constant     uint&  D_out [[buffer(5)]],
    constant     uint&  D_in [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint j = gid.y;
    if (i >= D_out || j >= D_in) {
        return;
    }
    float v = W[i * D_in + j] + lr * eps[i] * r[j];
    dst[i * D_in + j] = v;
}
