#include <metal_stdlib>

using namespace metal;

// Matches pkg/backend/device/cpu/math.FastExp32 (platform scalar for SwiGLU).
static inline float metal_fast_exp32(float value) {
    if (value < -87.33654f) {
        return 0.0f;
    }

    if (value > 88.72283f) {
        return 0x1.fffffep127f;
    }

    float z = value * 1.4426950408889634f;
    int exponentK = int(z);

    if (z < 0.0f) {
        exponentK--;
    }

    float fraction = z - float(exponentK);
    float poly = 1.0f + fraction * (
        0.69314718f + fraction * (
            0.24022650f + fraction * (
                0.05550410f + fraction * (
                    0.00961812f + fraction * 0.00133389f
                )
            )
        )
    );

    uint bits = as_type<uint>(poly) + uint(exponentK) * 8388608u;
    return as_type<float>(bits);
}

static inline float4 metal_fast_exp32_float4(float4 value) {
    return float4(
        metal_fast_exp32(value.x),
        metal_fast_exp32(value.y),
        metal_fast_exp32(value.z),
        metal_fast_exp32(value.w)
    );
}

static inline float4 swiglu_float4(float4 gate, float4 up) {
    float4 silu = gate / (float4(1.0f) + metal_fast_exp32_float4(-gate));
    return silu * up;
}

static inline float swiglu_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort swiglu_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline void swiglu_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

static inline void swiglu_float32_kernel(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device float* destinationScalar = (device float*)destination;
    device const float* gateScalar = (device const float*)gateVector;
    device const float* upScalar = (device const float*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        destination[index] = swiglu_float4(gateVector[index], upVector[index]);
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        gateScalar[offset],
        gateScalar[offset + 1u],
        gateScalar[offset + 2u],
        gateScalar[offset + 3u]
    );
    float4 upVec = float4(
        upScalar[offset],
        upScalar[offset + 1u],
        upScalar[offset + 2u],
        upScalar[offset + 3u]
    );
    float4 result = swiglu_float4(gateVec, upVec);

    swiglu_write_tail(destinationScalar, offset, tail, result);
}

kernel void swiglu_float32(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    swiglu_float32_kernel(destination, gateVector, upVector, count, index);
}

static inline void swiglu_float16_kernel(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device half* destinationScalar = (device half*)destination;
    device const half* gateScalar = (device const half*)gateVector;
    device const half* upScalar = (device const half*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(gateVector[index]);
        float4 upVec = float4(upVector[index]);
        destination[index] = half4(swiglu_float4(gateVec, upVec));
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        float(gateScalar[offset]),
        float(gateScalar[offset + 1u]),
        float(gateScalar[offset + 2u]),
        float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        float(upScalar[offset]),
        float(upScalar[offset + 1u]),
        float(upScalar[offset + 2u]),
        float(upScalar[offset + 3u])
    );
    float4 result = swiglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = half(result[lane]);
    }
}

kernel void swiglu_float16(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    swiglu_float16_kernel(destination, gateVector, upVector, count, index);
}

static inline void swiglu_bfloat16_kernel(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device ushort* destinationScalar = (device ushort*)destination;
    device const ushort* gateScalar = (device const ushort*)gateVector;
    device const ushort* upScalar = (device const ushort*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(
            swiglu_bf16_to_float(gateVector[index].x),
            swiglu_bf16_to_float(gateVector[index].y),
            swiglu_bf16_to_float(gateVector[index].z),
            swiglu_bf16_to_float(gateVector[index].w)
        );
        float4 upVec = float4(
            swiglu_bf16_to_float(upVector[index].x),
            swiglu_bf16_to_float(upVector[index].y),
            swiglu_bf16_to_float(upVector[index].z),
            swiglu_bf16_to_float(upVector[index].w)
        );
        float4 result = swiglu_float4(gateVec, upVec);
        destination[index] = ushort4(
            swiglu_float_to_bf16(result.x),
            swiglu_float_to_bf16(result.y),
            swiglu_float_to_bf16(result.z),
            swiglu_float_to_bf16(result.w)
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        swiglu_bf16_to_float(gateScalar[offset]),
        swiglu_bf16_to_float(gateScalar[offset + 1u]),
        swiglu_bf16_to_float(gateScalar[offset + 2u]),
        swiglu_bf16_to_float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        swiglu_bf16_to_float(upScalar[offset]),
        swiglu_bf16_to_float(upScalar[offset + 1u]),
        swiglu_bf16_to_float(upScalar[offset + 2u]),
        swiglu_bf16_to_float(upScalar[offset + 3u])
    );
    float4 result = swiglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = swiglu_float_to_bf16(result[lane]);
    }
}

kernel void swiglu_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    swiglu_bfloat16_kernel(destination, gateVector, upVector, count, index);
}
