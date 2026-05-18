#include <metal_stdlib>

using namespace metal;

kernel void add_float32(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    out[index] = left[index] + right[index];
}
