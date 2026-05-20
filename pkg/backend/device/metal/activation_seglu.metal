#include <metal_stdlib>

using namespace metal;

// Matches pkg/backend/device/cpu/math.FastExp32 (platform scalar for SeGLU).
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

// Matches pkg/backend/device/cpu/math.FastSigmoid32 (platform scalar for SeGLU).
static inline float metal_fast_sigmoid32(float value) {
    if (value >= 0.0f) {
        return 1.0f / (1.0f + metal_fast_exp32(-value));
    }

    float expValue = metal_fast_exp32(value);
    return expValue / (1.0f + expValue);
}

static inline float4 metal_fast_sigmoid32_float4(float4 value) {
    return float4(
        metal_fast_sigmoid32(value.x),
        metal_fast_sigmoid32(value.y),
        metal_fast_sigmoid32(value.z),
        metal_fast_sigmoid32(value.w)
    );
}

// up * FastSigmoid32(gate) — matches pkg/backend/device/cpu/math.FastSeGLU32.
static inline float4 seglu_float4(float4 gate, float4 up) {
    return up * metal_fast_sigmoid32_float4(gate);
}

static inline float seglu_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort seglu_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline void seglu_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

static inline void seglu_float32_kernel(
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
        destination[index] = seglu_float4(gateVector[index], upVector[index]);
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
    float4 result = seglu_float4(gateVec, upVec);

    seglu_write_tail(destinationScalar, offset, tail, result);
}

kernel void seglu_float32(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    seglu_float32_kernel(destination, gateVector, upVector, count, index);
}

static inline void seglu_float16_kernel(
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
        destination[index] = half4(seglu_float4(gateVec, upVec));
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
    float4 result = seglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = half(result[lane]);
    }
}

kernel void seglu_float16(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    seglu_float16_kernel(destination, gateVector, upVector, count, index);
}

static inline void seglu_bfloat16_kernel(
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
            seglu_bf16_to_float(gateVector[index].x),
            seglu_bf16_to_float(gateVector[index].y),
            seglu_bf16_to_float(gateVector[index].z),
            seglu_bf16_to_float(gateVector[index].w)
        );
        float4 upVec = float4(
            seglu_bf16_to_float(upVector[index].x),
            seglu_bf16_to_float(upVector[index].y),
            seglu_bf16_to_float(upVector[index].z),
            seglu_bf16_to_float(upVector[index].w)
        );
        float4 result = seglu_float4(gateVec, upVec);
        destination[index] = ushort4(
            seglu_float_to_bf16(result.x),
            seglu_float_to_bf16(result.y),
            seglu_float_to_bf16(result.z),
            seglu_float_to_bf16(result.w)
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        seglu_bf16_to_float(gateScalar[offset]),
        seglu_bf16_to_float(gateScalar[offset + 1u]),
        seglu_bf16_to_float(gateScalar[offset + 2u]),
        seglu_bf16_to_float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        seglu_bf16_to_float(upScalar[offset]),
        seglu_bf16_to_float(upScalar[offset + 1u]),
        seglu_bf16_to_float(upScalar[offset + 2u]),
        seglu_bf16_to_float(upScalar[offset + 3u])
    );
    float4 result = seglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = seglu_float_to_bf16(result[lane]);
    }
}

kernel void seglu_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    seglu_bfloat16_kernel(destination, gateVector, upVector, count, index);
}
