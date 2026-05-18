#include <metal_stdlib>

using namespace metal;

constant uint normalizationThreadCount = 256;
constant float layerNormEpsilonMetal = 1.0e-5f;
constant float rmsNormEpsilonMetal = 1.0e-6f;

static inline float bf16_to_float_norm(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort float_to_bf16_norm(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32NormStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16NormStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16NormStorage {
    static float load(device const ushort* values, uint index) {
        return bf16_to_float_norm(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = float_to_bf16_norm(value);
    }
};

template <typename Storage, typename Scalar>
static inline float reduce_sum(
    device const Scalar* input,
    threadgroup float* reduction,
    uint rowOffset,
    uint cols,
    uint threadIndex
) {
    float localSum = 0.0f;
    float localCompensation = 0.0f;

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float value = Storage::load(input, rowOffset + col) - localCompensation;
        float nextSum = localSum + value;
        localCompensation = (nextSum - localSum) - value;
        localSum = nextSum;
    }

    reduction[threadIndex] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    return reduction[0];
}

template <typename Storage, typename Scalar>
static inline void layernorm_rows(
    device const Scalar* input,
    device const Scalar* scale,
    device const Scalar* bias,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& cols,
    uint row,
    uint threadIndex
) {
    uint rowOffset = row * cols;
    float mean = reduce_sum<Storage, Scalar>(input, reduction, rowOffset, cols, threadIndex) /
        float(cols);
    float localVariance = 0.0f;
    float localCompensation = 0.0f;

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float delta = Storage::load(input, rowOffset + col) - mean;
        float value = delta * delta - localCompensation;
        float nextVariance = localVariance + value;
        localCompensation = (nextVariance - localVariance) - value;
        localVariance = nextVariance;
    }

    reduction[threadIndex] = localVariance;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float invStdDev = 1.0f / sqrt(reduction[0] / float(cols) + layerNormEpsilonMetal);

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float normalized = (Storage::load(input, rowOffset + col) - mean) * invStdDev;
        float value = normalized * Storage::load(scale, col) + Storage::load(bias, col);
        Storage::store(out, rowOffset + col, value);
    }
}

template <typename Storage, typename Scalar>
static inline void rmsnorm_rows(
    device const Scalar* input,
    device const Scalar* scale,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& cols,
    uint row,
    uint threadIndex
) {
    uint rowOffset = row * cols;
    float localSquareSum = 0.0f;
    float localCompensation = 0.0f;

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float value = Storage::load(input, rowOffset + col);
        float square = value * value - localCompensation;
        float nextSum = localSquareSum + square;
        localCompensation = (nextSum - localSquareSum) - square;
        localSquareSum = nextSum;
    }

    reduction[threadIndex] = localSquareSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float invRMS = 1.0f / sqrt(reduction[0] / float(cols) + rmsNormEpsilonMetal);

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float value = Storage::load(input, rowOffset + col) * invRMS * Storage::load(scale, col);
        Storage::store(out, rowOffset + col, value);
    }
}

#define LAYERNORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* scale [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& cols [[buffer(4)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    layernorm_rows<storage, scalar>(input, scale, bias, out, reduction, cols, row, threadIndex); \
}

#define RMSNORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* scale [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& cols [[buffer(3)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    rmsnorm_rows<storage, scalar>(input, scale, out, reduction, cols, row, threadIndex); \
}

kernel void layernorm_float32(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    threadgroup float stats[2];
    uint rowOffset = row * cols;

    if (threadIndex == 0) {
        float sum = 0.0f;

        for (uint col = 0; col < cols; col++) {
            sum += input[rowOffset + col];
        }

        float mean = sum / float(cols);
        float variance = 0.0f;

        for (uint col = 0; col < cols; col++) {
            float delta = input[rowOffset + col] - mean;
            variance += delta * delta;
        }

        stats[0] = mean;
        stats[1] = 1.0f / sqrt(variance / float(cols) + layerNormEpsilonMetal);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = stats[0];
    float invStdDev = stats[1];

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float normalized = (input[rowOffset + col] - mean) * invStdDev;
        out[rowOffset + col] = normalized * scale[col] + bias[col];
    }
}

LAYERNORM_KERNEL(layernorm_float16, Float16NormStorage, half)
LAYERNORM_KERNEL(layernorm_bfloat16, BFloat16NormStorage, ushort)

RMSNORM_KERNEL(rmsnorm_float32, Float32NormStorage, float)
RMSNORM_KERNEL(rmsnorm_float16, Float16NormStorage, half)
RMSNORM_KERNEL(rmsnorm_bfloat16, BFloat16NormStorage, ushort)
