#include <metal_stdlib>

using namespace metal;

constant uint hawkesMarkovThreadCount = 256;

static inline float hawkes_markov_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort hawkes_markov_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline float hawkes_markov_safe_positive(float value) {
    return max(value, 1.0e-12f);
}

struct Float32HawkesMarkovStorage {
    static float load(device const float* values, uint index) { return values[index]; }
    static void store(device float* values, uint index, float value) { values[index] = value; }
};

struct Float16HawkesMarkovStorage {
    static float load(device const half* values, uint index) { return float(values[index]); }
    static void store(device half* values, uint index, float value) { values[index] = half(value); }
};

struct BFloat16HawkesMarkovStorage {
    static float load(device const ushort* values, uint index) {
        return hawkes_markov_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = hawkes_markov_float_to_bf16(value);
    }
};

static inline float hawkes_markov_reduce(threadgroup float* reduction, uint threadIndex) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = hawkesMarkovThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    return reduction[0];
}

template <typename Storage, typename Scalar>
static inline void hawkes_intensity_kernel(
    device const Scalar* events,
    device const Scalar* queryTimes,
    device const Scalar* baseline,
    device const Scalar* alpha,
    device const Scalar* beta,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& eventCount,
    uint queryIndex,
    uint threadIndex
) {
    float queryTime = Storage::load(queryTimes, queryIndex);
    float alphaValue = Storage::load(alpha, 0);
    float betaValue = Storage::load(beta, 0);
    float localValue = 0.0f;

    for (uint eventIndex = threadIndex; eventIndex < eventCount; eventIndex += hawkesMarkovThreadCount) {
        float eventTime = Storage::load(events, eventIndex);
        if (eventTime <= queryTime) {
            localValue += alphaValue * exp(-betaValue * (queryTime - eventTime));
        }
    }

    reduction[threadIndex] = localValue;
    float sum = hawkes_markov_reduce(reduction, threadIndex);

    if (threadIndex == 0) {
        Storage::store(out, queryIndex, Storage::load(baseline, 0) + sum);
    }
}

template <typename Storage, typename Scalar>
static inline void hawkes_kernel_matrix_kernel(
    device const Scalar* events,
    device const Scalar* alpha,
    device const Scalar* beta,
    device Scalar* out,
    constant uint& eventCount,
    uint index
) {
    uint total = eventCount * eventCount;
    if (index >= total) {
        return;
    }

    uint row = index / eventCount;
    uint col = index - row * eventCount;
    float value = 0.0f;

    if (col < row) {
        float delta = Storage::load(events, row) - Storage::load(events, col);
        value = Storage::load(alpha, 0) * exp(-Storage::load(beta, 0) * delta);
    }

    Storage::store(out, index, value);
}

template <typename Storage, typename Scalar>
static inline void hawkes_log_likelihood_partial_kernel(
    device const Scalar* events,
    device const Scalar* totalTime,
    device const Scalar* baseline,
    device const Scalar* alpha,
    device const Scalar* beta,
    device float* scratch,
    threadgroup float* reduction,
    constant uint& eventCount,
    uint groupIndex,
    uint threadIndex
) {
    uint eventIndex = groupIndex * hawkesMarkovThreadCount + threadIndex;
    float localValue = 0.0f;

    if (eventIndex < eventCount) {
        float eventTime = Storage::load(events, eventIndex);
        float mu = Storage::load(baseline, 0);
        float alphaValue = Storage::load(alpha, 0);
        float betaValue = Storage::load(beta, 0);
        float intensity = mu;

        for (uint previousIndex = 0; previousIndex < eventIndex; previousIndex++) {
            float delta = eventTime - Storage::load(events, previousIndex);
            intensity += alphaValue * exp(-betaValue * delta);
        }

        float compensator = (alphaValue / betaValue) *
            (1.0f - exp(-betaValue * (Storage::load(totalTime, 0) - eventTime)));
        localValue = log(hawkes_markov_safe_positive(intensity)) - compensator;
    }

    reduction[threadIndex] = localValue;
    float sum = hawkes_markov_reduce(reduction, threadIndex);

    if (threadIndex == 0) {
        scratch[groupIndex] = sum;
    }
}

template <typename Storage, typename Scalar>
static inline void hawkes_log_likelihood_finalize_kernel(
    device const float* scratch,
    device const Scalar* totalTime,
    device const Scalar* baseline,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& partialCount,
    uint threadIndex
) {
    float localValue = 0.0f;

    for (uint index = threadIndex; index < partialCount; index += hawkesMarkovThreadCount) {
        localValue += scratch[index];
    }

    reduction[threadIndex] = localValue;
    float sum = hawkes_markov_reduce(reduction, threadIndex);

    if (threadIndex == 0) {
        Storage::store(out, 0, sum - Storage::load(baseline, 0) * Storage::load(totalTime, 0));
    }
}

template <typename Storage, typename Scalar>
static inline void markov_mutual_information_partial_kernel(
    device const Scalar* joint,
    device float* scratch,
    threadgroup float* reduction,
    constant uint& rows,
    constant uint& cols,
    uint groupIndex,
    uint threadIndex
) {
    uint total = rows * cols;
    uint flatIndex = groupIndex * hawkesMarkovThreadCount + threadIndex;
    float localValue = 0.0f;

    if (flatIndex < total) {
        uint row = flatIndex / cols;
        uint col = flatIndex - row * cols;
        float jointValue = Storage::load(joint, flatIndex);

        if (jointValue > 1.0e-12f) {
            float marginalRow = 0.0f;
            float marginalCol = 0.0f;

            for (uint colIndex = 0; colIndex < cols; colIndex++) {
                marginalRow += Storage::load(joint, row * cols + colIndex);
            }

            for (uint rowIndex = 0; rowIndex < rows; rowIndex++) {
                marginalCol += Storage::load(joint, rowIndex * cols + col);
            }

            localValue = jointValue * log(jointValue / (marginalRow * marginalCol + 1.0e-12f));
        }
    }

    reduction[threadIndex] = localValue;
    float sum = hawkes_markov_reduce(reduction, threadIndex);

    if (threadIndex == 0) {
        scratch[groupIndex] = sum;
    }
}

template <typename Storage, typename Scalar>
static inline void hawkes_markov_finalize_kernel(
    device const float* scratch,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& partialCount,
    uint threadIndex
) {
    float localValue = 0.0f;

    for (uint index = threadIndex; index < partialCount; index += hawkesMarkovThreadCount) {
        localValue += scratch[index];
    }

    reduction[threadIndex] = localValue;
    float sum = hawkes_markov_reduce(reduction, threadIndex);

    if (threadIndex == 0) {
        Storage::store(out, 0, sum);
    }
}

template <typename Storage, typename Scalar>
static inline void markov_blanket_partition_kernel(
    device const Scalar* adjacency,
    device const int* internalNodes,
    device int* out,
    constant uint& nodeCount,
    constant uint& internalCount,
    uint nodeIndex
) {
    if (nodeIndex >= nodeCount) {
        return;
    }

    bool isInternal = false;
    bool incomingFromInternal = false;
    bool outgoingToInternal = false;

    for (uint index = 0; index < internalCount; index++) {
        int internalNode = internalNodes[index];
        if (internalNode < 0 || uint(internalNode) >= nodeCount) {
            continue;
        }

        uint other = uint(internalNode);
        isInternal = isInternal || other == nodeIndex;
        incomingFromInternal = incomingFromInternal || Storage::load(adjacency, other * nodeCount + nodeIndex) != 0.0f;
        outgoingToInternal = outgoingToInternal || Storage::load(adjacency, nodeIndex * nodeCount + other) != 0.0f;
    }

    if (isInternal) {
        out[nodeIndex] = 0;
        return;
    }

    if (incomingFromInternal && outgoingToInternal) {
        out[nodeIndex] = 2;
        return;
    }

    if (outgoingToInternal) {
        out[nodeIndex] = 1;
        return;
    }

    out[nodeIndex] = 3;
}

template <typename Storage, typename Scalar>
static inline void markov_flow_kernel(
    device const Scalar* mutualInformation,
    device const int* partition,
    device Scalar* out,
    constant uint& nodeCount,
    constant int& targetLabel,
    uint nodeIndex
) {
    if (nodeIndex >= nodeCount) {
        return;
    }

    if (partition[nodeIndex] != targetLabel) {
        Storage::store(out, nodeIndex, 0.0f);
        return;
    }

    float sum = 0.0f;
    for (uint otherIndex = 0; otherIndex < nodeCount; otherIndex++) {
        if (partition[otherIndex] == 0) {
            sum += Storage::load(mutualInformation, nodeIndex * nodeCount + otherIndex);
        }
    }

    Storage::store(out, nodeIndex, sum);
}

#define HAWKES_INTENSITY_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* events [[buffer(0)]], \
    device const scalar* queryTimes [[buffer(1)]], \
    device const scalar* baseline [[buffer(2)]], \
    device const scalar* alpha [[buffer(3)]], \
    device const scalar* beta [[buffer(4)]], \
    device scalar* out [[buffer(5)]], \
    constant uint& eventCount [[buffer(6)]], \
    uint queryIndex [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    hawkes_intensity_kernel<storage, scalar>( \
        events, queryTimes, baseline, alpha, beta, out, reduction, \
        eventCount, queryIndex, threadIndex \
    ); \
}

#define HAWKES_KERNEL_MATRIX_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* events [[buffer(0)]], \
    device const scalar* alpha [[buffer(1)]], \
    device const scalar* beta [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& eventCount [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    hawkes_kernel_matrix_kernel<storage, scalar>(events, alpha, beta, out, eventCount, index); \
}

#define HAWKES_LOG_PARTIAL_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* events [[buffer(0)]], \
    device const scalar* totalTime [[buffer(1)]], \
    device const scalar* baseline [[buffer(2)]], \
    device const scalar* alpha [[buffer(3)]], \
    device const scalar* beta [[buffer(4)]], \
    device float* scratch [[buffer(5)]], \
    constant uint& eventCount [[buffer(6)]], \
    uint groupIndex [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    hawkes_log_likelihood_partial_kernel<storage, scalar>( \
        events, totalTime, baseline, alpha, beta, scratch, reduction, eventCount, groupIndex, threadIndex \
    ); \
}

#define HAWKES_LOG_FINALIZE_KERNEL(name, storage, scalar) \
kernel void name( \
    device const float* scratch [[buffer(0)]], \
    device const scalar* totalTime [[buffer(1)]], \
    device const scalar* baseline [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& partialCount [[buffer(4)]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    hawkes_log_likelihood_finalize_kernel<storage, scalar>( \
        scratch, totalTime, baseline, out, reduction, partialCount, threadIndex \
    ); \
}

#define MARKOV_MI_PARTIAL_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* joint [[buffer(0)]], \
    device float* scratch [[buffer(1)]], \
    constant uint& rows [[buffer(2)]], \
    constant uint& cols [[buffer(3)]], \
    uint groupIndex [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    markov_mutual_information_partial_kernel<storage, scalar>( \
        joint, scratch, reduction, rows, cols, groupIndex, threadIndex \
    ); \
}

#define HAWKES_MARKOV_FINALIZE_KERNEL(name, storage, scalar) \
kernel void name( \
    device const float* scratch [[buffer(0)]], \
    device scalar* out [[buffer(1)]], \
    constant uint& partialCount [[buffer(2)]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    hawkes_markov_finalize_kernel<storage, scalar>(scratch, out, reduction, partialCount, threadIndex); \
}

#define MARKOV_PARTITION_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* adjacency [[buffer(0)]], \
    device const int* internalNodes [[buffer(1)]], \
    device int* out [[buffer(2)]], \
    constant uint& nodeCount [[buffer(3)]], \
    constant uint& internalCount [[buffer(4)]], \
    uint nodeIndex [[thread_position_in_grid]] \
) { \
    markov_blanket_partition_kernel<storage, scalar>( \
        adjacency, internalNodes, out, nodeCount, internalCount, nodeIndex \
    ); \
}

#define MARKOV_FLOW_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* mutualInformation [[buffer(0)]], \
    device const int* partition [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& nodeCount [[buffer(3)]], \
    constant int& targetLabel [[buffer(4)]], \
    uint nodeIndex [[thread_position_in_grid]] \
) { \
    markov_flow_kernel<storage, scalar>( \
        mutualInformation, partition, out, nodeCount, targetLabel, nodeIndex \
    ); \
}

HAWKES_INTENSITY_KERNEL(hawkes_intensity_float32, Float32HawkesMarkovStorage, float)
HAWKES_KERNEL_MATRIX_KERNEL(hawkes_kernel_matrix_float32, Float32HawkesMarkovStorage, float)
HAWKES_LOG_PARTIAL_KERNEL(hawkes_log_likelihood_float32_partial, Float32HawkesMarkovStorage, float)
HAWKES_LOG_FINALIZE_KERNEL(hawkes_log_likelihood_float32_finalize, Float32HawkesMarkovStorage, float)
MARKOV_MI_PARTIAL_KERNEL(markov_mutual_information_float32_partial, Float32HawkesMarkovStorage, float)
HAWKES_MARKOV_FINALIZE_KERNEL(hawkes_markov_finalize_float32, Float32HawkesMarkovStorage, float)
MARKOV_PARTITION_KERNEL(markov_blanket_partition_float32, Float32HawkesMarkovStorage, float)
MARKOV_FLOW_KERNEL(markov_flow_float32, Float32HawkesMarkovStorage, float)

HAWKES_INTENSITY_KERNEL(hawkes_intensity_float16, Float16HawkesMarkovStorage, half)
HAWKES_KERNEL_MATRIX_KERNEL(hawkes_kernel_matrix_float16, Float16HawkesMarkovStorage, half)
HAWKES_LOG_PARTIAL_KERNEL(hawkes_log_likelihood_float16_partial, Float16HawkesMarkovStorage, half)
HAWKES_LOG_FINALIZE_KERNEL(hawkes_log_likelihood_float16_finalize, Float16HawkesMarkovStorage, half)
MARKOV_MI_PARTIAL_KERNEL(markov_mutual_information_float16_partial, Float16HawkesMarkovStorage, half)
HAWKES_MARKOV_FINALIZE_KERNEL(hawkes_markov_finalize_float16, Float16HawkesMarkovStorage, half)
MARKOV_PARTITION_KERNEL(markov_blanket_partition_float16, Float16HawkesMarkovStorage, half)
MARKOV_FLOW_KERNEL(markov_flow_float16, Float16HawkesMarkovStorage, half)

HAWKES_INTENSITY_KERNEL(hawkes_intensity_bfloat16, BFloat16HawkesMarkovStorage, ushort)
HAWKES_KERNEL_MATRIX_KERNEL(hawkes_kernel_matrix_bfloat16, BFloat16HawkesMarkovStorage, ushort)
HAWKES_LOG_PARTIAL_KERNEL(hawkes_log_likelihood_bfloat16_partial, BFloat16HawkesMarkovStorage, ushort)
HAWKES_LOG_FINALIZE_KERNEL(hawkes_log_likelihood_bfloat16_finalize, BFloat16HawkesMarkovStorage, ushort)
MARKOV_MI_PARTIAL_KERNEL(markov_mutual_information_bfloat16_partial, BFloat16HawkesMarkovStorage, ushort)
HAWKES_MARKOV_FINALIZE_KERNEL(hawkes_markov_finalize_bfloat16, BFloat16HawkesMarkovStorage, ushort)
MARKOV_PARTITION_KERNEL(markov_blanket_partition_bfloat16, BFloat16HawkesMarkovStorage, ushort)
MARKOV_FLOW_KERNEL(markov_flow_bfloat16, BFloat16HawkesMarkovStorage, ushort)
