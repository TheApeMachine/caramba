#include <metal_stdlib>
#include "elementwise_gelu_f64.metalinc"

using namespace metal;

// gate * FastGelu32(up) — matches pkg/backend/device/cpu/math.FastGeGLU32.
static inline float4 geglu_float4(float4 gate, float4 up) {
    return gate * metal_gelu_float4(up);
}

static inline float geglu_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort geglu_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline void geglu_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

static inline void geglu_float32_kernel(
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
        destination[index] = geglu_float4(gateVector[index], upVector[index]);
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
    float4 result = geglu_float4(gateVec, upVec);

    geglu_write_tail(destinationScalar, offset, tail, result);
}

kernel void geglu_float32(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    geglu_float32_kernel(destination, gateVector, upVector, count, index);
}

static inline void geglu_float16_kernel(
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
        destination[index] = half4(geglu_float4(gateVec, upVec));
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
    float4 result = geglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = half(result[lane]);
    }
}

kernel void geglu_float16(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    geglu_float16_kernel(destination, gateVector, upVector, count, index);
}

static inline void geglu_bfloat16_kernel(
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
            geglu_bf16_to_float(gateVector[index].x),
            geglu_bf16_to_float(gateVector[index].y),
            geglu_bf16_to_float(gateVector[index].z),
            geglu_bf16_to_float(gateVector[index].w)
        );
        float4 upVec = float4(
            geglu_bf16_to_float(upVector[index].x),
            geglu_bf16_to_float(upVector[index].y),
            geglu_bf16_to_float(upVector[index].z),
            geglu_bf16_to_float(upVector[index].w)
        );
        float4 result = geglu_float4(gateVec, upVec);
        destination[index] = ushort4(
            geglu_float_to_bf16(result.x),
            geglu_float_to_bf16(result.y),
            geglu_float_to_bf16(result.z),
            geglu_float_to_bf16(result.w)
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        geglu_bf16_to_float(gateScalar[offset]),
        geglu_bf16_to_float(gateScalar[offset + 1u]),
        geglu_bf16_to_float(gateScalar[offset + 2u]),
        geglu_bf16_to_float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        geglu_bf16_to_float(upScalar[offset]),
        geglu_bf16_to_float(upScalar[offset + 1u]),
        geglu_bf16_to_float(upScalar[offset + 2u]),
        geglu_bf16_to_float(upScalar[offset + 3u])
    );
    float4 result = geglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = geglu_float_to_bf16(result[lane]);
    }
}

kernel void geglu_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    geglu_bfloat16_kernel(destination, gateVector, upVector, count, index);
}
