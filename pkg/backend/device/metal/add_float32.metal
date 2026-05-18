#include <metal_stdlib>

using namespace metal;

kernel void add_float32(
    device const float4* leftVector [[buffer(0)]],
    device const float4* rightVector [[buffer(1)]],
    device float4* outVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 4;

    if (base + 3 < count) {
        outVector[index] = leftVector[index] + rightVector[index];
        return;
    }

    device const float* left = reinterpret_cast<device const float*>(leftVector);
    device const float* right = reinterpret_cast<device const float*>(rightVector);
    device float* out = reinterpret_cast<device float*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = left[scalarIndex] + right[scalarIndex];
        }
    }
}
