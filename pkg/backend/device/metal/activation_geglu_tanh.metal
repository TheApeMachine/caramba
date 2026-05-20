#include <metal_stdlib>

using namespace metal;

// pkg/backend/device/cpu/math/constant.go GeluTanhAlpha, GeluTanhBeta.
constant float gegluTanhAlpha = 0.7978845608028654f;
constant float gegluTanhBeta = 0.044715f;

// Padé tanh — matches pkg/backend/device/cpu/math.FastTanh32.
static inline float metal_fast_tanh32(float value) {
    if (value > 4.92f) {
        return 1.0f;
    }

    if (value < -4.92f) {
        return -1.0f;
    }

    float valueSquared = value * value;
    float numerator = value * (
        135135.0f + valueSquared * (17325.0f + valueSquared * (378.0f + valueSquared))
    );
    float denominator = 135135.0f + valueSquared * (
        62370.0f + valueSquared * (3150.0f + valueSquared * 28.0f)
    );

    return numerator / denominator;
}

// tanh GELU — matches pkg/backend/device/cpu/math.FastGeluTanh32 (per-lane f32).
static inline float metal_fast_gelu_tanh32(float value) {
    float cubic = value * value * value;
    float inner = gegluTanhAlpha * (value + gegluTanhBeta * cubic);

    return 0.5f * value * (1.0f + metal_fast_tanh32(inner));
}

static inline float4 metal_fast_gelu_tanh32_float4(float4 value) {
    return float4(
        metal_fast_gelu_tanh32(value.x),
        metal_fast_gelu_tanh32(value.y),
        metal_fast_gelu_tanh32(value.z),
        metal_fast_gelu_tanh32(value.w)
    );
}

// gate * FastGeluTanh32(up) — matches pkg/backend/device/cpu/math.FastGeGLUTanh32.
static inline float4 geglu_tanh_float4(float4 gate, float4 up) {
    return gate * metal_fast_gelu_tanh32_float4(up);
}

static inline float geglu_tanh_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort geglu_tanh_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline void geglu_tanh_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

static inline void geglu_tanh_float32_kernel(
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
        destination[index] = geglu_tanh_float4(gateVector[index], upVector[index]);
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
    float4 result = geglu_tanh_float4(gateVec, upVec);

    geglu_tanh_write_tail(destinationScalar, offset, tail, result);
}

kernel void geglu_tanh_float32(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    geglu_tanh_float32_kernel(destination, gateVector, upVector, count, index);
}

static inline void geglu_tanh_float16_kernel(
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
        destination[index] = half4(geglu_tanh_float4(gateVec, upVec));
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
    float4 result = geglu_tanh_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = half(result[lane]);
    }
}

kernel void geglu_tanh_float16(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    geglu_tanh_float16_kernel(destination, gateVector, upVector, count, index);
}

static inline void geglu_tanh_bfloat16_kernel(
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
            geglu_tanh_bf16_to_float(gateVector[index].x),
            geglu_tanh_bf16_to_float(gateVector[index].y),
            geglu_tanh_bf16_to_float(gateVector[index].z),
            geglu_tanh_bf16_to_float(gateVector[index].w)
        );
        float4 upVec = float4(
            geglu_tanh_bf16_to_float(upVector[index].x),
            geglu_tanh_bf16_to_float(upVector[index].y),
            geglu_tanh_bf16_to_float(upVector[index].z),
            geglu_tanh_bf16_to_float(upVector[index].w)
        );
        float4 result = geglu_tanh_float4(gateVec, upVec);
        destination[index] = ushort4(
            geglu_tanh_float_to_bf16(result.x),
            geglu_tanh_float_to_bf16(result.y),
            geglu_tanh_float_to_bf16(result.z),
            geglu_tanh_float_to_bf16(result.w)
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        geglu_tanh_bf16_to_float(gateScalar[offset]),
        geglu_tanh_bf16_to_float(gateScalar[offset + 1u]),
        geglu_tanh_bf16_to_float(gateScalar[offset + 2u]),
        geglu_tanh_bf16_to_float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        geglu_tanh_bf16_to_float(upScalar[offset]),
        geglu_tanh_bf16_to_float(upScalar[offset + 1u]),
        geglu_tanh_bf16_to_float(upScalar[offset + 2u]),
        geglu_tanh_bf16_to_float(upScalar[offset + 3u])
    );
    float4 result = geglu_tanh_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = geglu_tanh_float_to_bf16(result[lane]);
    }
}

kernel void geglu_tanh_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    geglu_tanh_bfloat16_kernel(destination, gateVector, upVector, count, index);
}
