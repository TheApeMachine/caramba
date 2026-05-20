#include <metal_stdlib>
using namespace metal;

static inline void weight_graft_write_tail(
    device float* weights,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        weights[offset + lane] = result[lane];
    }
}

kernel void weight_graft_add_float32(
    device float4* weights [[buffer(0)]],
    device const float4* injection [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    device float* weightsScalar = (device float*)weights;
    device const float* injectionScalar = (device const float*)injection;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        weights[index] = weights[index] + injection[index];
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 weightVec = float4(
        weightsScalar[offset],
        weightsScalar[offset + 1u],
        weightsScalar[offset + 2u],
        weightsScalar[offset + 3u]
    );
    float4 injectionVec = float4(
        injectionScalar[offset],
        injectionScalar[offset + 1u],
        injectionScalar[offset + 2u],
        injectionScalar[offset + 3u]
    );
    float4 result = weightVec + injectionVec;

    weight_graft_write_tail(weightsScalar, offset, tail, result);
}
