#include <metal_stdlib>
using namespace metal;

static inline float4 activation_steer_float4(
    float4 baseVector,
    float4 directionVector,
    float coefficient
) {
    return baseVector + coefficient * directionVector;
}

static inline void activation_steer_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

kernel void activation_steer_float32(
    device float4* destination [[buffer(0)]],
    device const float4* baseVector [[buffer(1)]],
    device const float4* directionVector [[buffer(2)]],
    constant float& coefficient [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    device float* destinationScalar = (device float*)destination;
    device const float* baseScalar = (device const float*)baseVector;
    device const float* directionScalar = (device const float*)directionVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        destination[index] = activation_steer_float4(
            baseVector[index],
            directionVector[index],
            coefficient
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 baseVec = float4(
        baseScalar[offset],
        baseScalar[offset + 1u],
        baseScalar[offset + 2u],
        baseScalar[offset + 3u]
    );
    float4 directionVec = float4(
        directionScalar[offset],
        directionScalar[offset + 1u],
        directionScalar[offset + 2u],
        directionScalar[offset + 3u]
    );
    float4 result = activation_steer_float4(baseVec, directionVec, coefficient);

    activation_steer_write_tail(destinationScalar, offset, tail, result);
}
