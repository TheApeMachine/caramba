#include <metal_stdlib>

using namespace metal;

constant uint reductionThreadCount = 256;

static inline float reduction_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort reduction_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32ReductionStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16ReductionStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16ReductionStorage {
    static float load(device const ushort* values, uint index) {
        return reduction_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = reduction_float_to_bf16(value);
    }
};

static inline bool reduction_is_sum_like(uint operation) {
    return operation == 0u || operation == 1u || operation == 7u ||
        operation == 8u || operation == 9u || operation == 10u;
}

static inline bool reduction_is_arg(uint operation) {
    return operation == 5u || operation == 6u;
}

static inline float reduction_identity_a(uint operation) {
    if (operation == 2u) {
        return 1.0f;
    }

    if (operation == 3u || operation == 5u) {
        return 3.4028234663852886e38f;
    }

    if (operation == 4u || operation == 6u) {
        return -3.4028234663852886e38f;
    }

    return 0.0f;
}

static inline float reduction_partial_a(float value, uint operation) {
    if (operation == 7u) {
        return abs(value);
    }

    if (operation == 8u || operation == 9u || operation == 10u) {
        return value * value;
    }

    return value;
}

static inline float reduction_partial_b(float value, uint operation) {
    if (operation == 9u || operation == 10u) {
        return value;
    }

    return 0.0f;
}

static inline void reduction_combine_arg(
    threadgroup float* reductionA,
    threadgroup float* reductionB,
    uint leftIndex,
    uint rightIndex,
    bool useMax
) {
    float leftValue = reductionA[leftIndex];
    float rightValue = reductionA[rightIndex];
    bool takeRight = useMax ? rightValue > leftValue : rightValue < leftValue;

    if (!takeRight) {
        return;
    }

    reductionA[leftIndex] = rightValue;
    reductionB[leftIndex] = reductionB[rightIndex];
}

static inline void reduction_combine_pair(
    threadgroup float* reductionA,
    threadgroup float* reductionB,
    uint operation,
    uint leftIndex,
    uint rightIndex
) {
    if (operation == 2u) {
        reductionA[leftIndex] *= reductionA[rightIndex];
        return;
    }

    if (operation == 3u) {
        reductionA[leftIndex] = min(reductionA[leftIndex], reductionA[rightIndex]);
        return;
    }

    if (operation == 4u) {
        reductionA[leftIndex] = max(reductionA[leftIndex], reductionA[rightIndex]);
        return;
    }

    if (operation == 5u || operation == 6u) {
        reduction_combine_arg(reductionA, reductionB, leftIndex, rightIndex, operation == 6u);
        return;
    }

    reductionA[leftIndex] += reductionA[rightIndex];
    reductionB[leftIndex] += reductionB[rightIndex];
}

template <typename Storage, typename Scalar>
static inline void reduction_partial(
    device const Scalar* input,
    device float* scratchA,
    device float* scratchB,
    threadgroup float* reductionA,
    threadgroup float* reductionB,
    constant uint& count,
    constant uint& operation,
    uint groupIndex,
    uint threadIndex
) {
    uint valueIndex = groupIndex * reductionThreadCount + threadIndex;
    float localA = reduction_identity_a(operation);
    float localB = 0.0f;

    if (valueIndex < count) {
        float value = Storage::load(input, valueIndex);
        localA = reduction_partial_a(value, operation);
        localB = reduction_is_arg(operation) ? float(valueIndex) : reduction_partial_b(value, operation);
    }

    reductionA[threadIndex] = localA;
    reductionB[threadIndex] = localB;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = reductionThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction_combine_pair(reductionA, reductionB, operation, threadIndex, threadIndex + stride);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        scratchA[groupIndex] = reductionA[0];
        scratchB[groupIndex] = reductionB[0];
    }
}

static inline float reduction_finalize_value(
    float accumulatedA,
    float accumulatedB,
    uint operation,
    uint count
) {
    if (operation == 1u) {
        return accumulatedA / float(count);
    }

    if (operation == 5u || operation == 6u) {
        return accumulatedB;
    }

    if (operation == 8u) {
        return sqrt(accumulatedA);
    }

    if (operation == 9u || operation == 10u) {
        float mean = accumulatedB / float(count);
        float variance = accumulatedA / float(count) - mean * mean;
        variance = max(variance, 0.0f);

        if (operation == 10u) {
            return sqrt(variance);
        }

        return variance;
    }

    return accumulatedA;
}

template <typename Storage, typename Scalar>
static inline void reduction_finalize(
    device const float* scratchA,
    device const float* scratchB,
    device Scalar* out,
    threadgroup float* reductionA,
    threadgroup float* reductionB,
    constant uint& partialCount,
    constant uint& count,
    constant uint& operation,
    uint threadIndex
) {
    float localA = reduction_identity_a(operation);
    float localB = 0.0f;

    if (reduction_is_sum_like(operation)) {
        localA = 0.0f;
    }

    for (uint index = threadIndex; index < partialCount; index += reductionThreadCount) {
        float candidateA = scratchA[index];
        float candidateB = scratchB[index];

        if (reduction_is_arg(operation)) {
            bool takeCandidate = operation == 6u ? candidateA > localA : candidateA < localA;
            if (takeCandidate) {
                localA = candidateA;
                localB = candidateB;
            }
            continue;
        }

        if (operation == 2u) {
            localA *= candidateA;
            continue;
        }

        if (operation == 3u) {
            localA = min(localA, candidateA);
            continue;
        }

        if (operation == 4u) {
            localA = max(localA, candidateA);
            continue;
        }

        localA += candidateA;
        localB += candidateB;
    }

    reductionA[threadIndex] = localA;
    reductionB[threadIndex] = localB;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = reductionThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction_combine_pair(reductionA, reductionB, operation, threadIndex, threadIndex + stride);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        Storage::store(
            out,
            0,
            reduction_finalize_value(reductionA[0], reductionB[0], operation, count)
        );
    }
}

#define REDUCTION_PARTIAL_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device float* scratchA [[buffer(1)]], \
    device float* scratchB [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    constant uint& operation [[buffer(4)]], \
    uint groupIndex [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reductionA[256]; \
    threadgroup float reductionB[256]; \
    reduction_partial<storage, scalar>( \
        input, scratchA, scratchB, reductionA, reductionB, count, operation, groupIndex, threadIndex \
    ); \
}

#define REDUCTION_FINALIZE_KERNEL(name, storage, scalar) \
kernel void name( \
    device const float* scratchA [[buffer(0)]], \
    device const float* scratchB [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& partialCount [[buffer(3)]], \
    constant uint& count [[buffer(4)]], \
    constant uint& operation [[buffer(5)]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reductionA[256]; \
    threadgroup float reductionB[256]; \
    reduction_finalize<storage, scalar>( \
        scratchA, scratchB, out, reductionA, reductionB, partialCount, count, operation, threadIndex \
    ); \
}

REDUCTION_PARTIAL_KERNEL(reduction_partial_float32, Float32ReductionStorage, float)
REDUCTION_FINALIZE_KERNEL(reduction_finalize_float32, Float32ReductionStorage, float)

REDUCTION_PARTIAL_KERNEL(reduction_partial_float16, Float16ReductionStorage, half)
REDUCTION_FINALIZE_KERNEL(reduction_finalize_float16, Float16ReductionStorage, half)

REDUCTION_PARTIAL_KERNEL(reduction_partial_bfloat16, BFloat16ReductionStorage, ushort)
REDUCTION_FINALIZE_KERNEL(reduction_finalize_bfloat16, BFloat16ReductionStorage, ushort)
