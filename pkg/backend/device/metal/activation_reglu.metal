#include <metal_stdlib>

using namespace metal;

// Matches pkg/backend/device/cpu/math.FastReLU32 (vector max for ReGLU).
static inline float4 metal_fast_relu32_float4(float4 value) {
    return max(value, 0.0f);
}

// gate * FastReLU32(up) — matches pkg/backend/device/cpu/math.FastReGLU32.
static inline float4 reglu_float4(float4 gate, float4 up) {
    return gate * metal_fast_relu32_float4(up);
}

static inline float reglu_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort reglu_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline void reglu_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

static inline void reglu_float32_kernel(
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
        destination[index] = reglu_float4(gateVector[index], upVector[index]);
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
    float4 result = reglu_float4(gateVec, upVec);

    reglu_write_tail(destinationScalar, offset, tail, result);
}

kernel void reglu_float32(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    reglu_float32_kernel(destination, gateVector, upVector, count, index);
}

static inline void reglu_float16_kernel(
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
        destination[index] = half4(reglu_float4(gateVec, upVec));
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
    float4 result = reglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = half(result[lane]);
    }
}

kernel void reglu_float16(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    reglu_float16_kernel(destination, gateVector, upVector, count, index);
}

static inline void reglu_bfloat16_kernel(
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
            reglu_bf16_to_float(gateVector[index].x),
            reglu_bf16_to_float(gateVector[index].y),
            reglu_bf16_to_float(gateVector[index].z),
            reglu_bf16_to_float(gateVector[index].w)
        );
        float4 upVec = float4(
            reglu_bf16_to_float(upVector[index].x),
            reglu_bf16_to_float(upVector[index].y),
            reglu_bf16_to_float(upVector[index].z),
            reglu_bf16_to_float(upVector[index].w)
        );
        float4 result = reglu_float4(gateVec, upVec);
        destination[index] = ushort4(
            reglu_float_to_bf16(result.x),
            reglu_float_to_bf16(result.y),
            reglu_float_to_bf16(result.z),
            reglu_float_to_bf16(result.w)
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        reglu_bf16_to_float(gateScalar[offset]),
        reglu_bf16_to_float(gateScalar[offset + 1u]),
        reglu_bf16_to_float(gateScalar[offset + 2u]),
        reglu_bf16_to_float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        reglu_bf16_to_float(upScalar[offset]),
        reglu_bf16_to_float(upScalar[offset + 1u]),
        reglu_bf16_to_float(upScalar[offset + 2u]),
        reglu_bf16_to_float(upScalar[offset + 3u])
    );
    float4 result = reglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = reglu_float_to_bf16(result[lane]);
    }
}

kernel void reglu_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    reglu_bfloat16_kernel(destination, gateVector, upVector, count, index);
}
