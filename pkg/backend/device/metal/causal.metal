#include <metal_stdlib>

using namespace metal;

constant uint causalThreadCount = 256;

static inline float causal_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort causal_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32CausalStorage {
    static float load(device const float* values, uint index) { return values[index]; }
    static void store(device float* values, uint index, float value) { values[index] = value; }
};

struct Float16CausalStorage {
    static float load(device const half* values, uint index) { return float(values[index]); }
    static void store(device half* values, uint index, float value) { values[index] = half(value); }
};

struct BFloat16CausalStorage {
    static float load(device const ushort* values, uint index) {
        return causal_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = causal_float_to_bf16(value);
    }
};

static inline float causal_safe_positive(float value) {
    return max(value, 1.0e-12f);
}

template <typename Storage, typename Scalar>
static inline void backdoor_adjustment_kernel(
    device const Scalar* conditional,
    device const Scalar* marginal,
    device Scalar* out,
    constant uint& xCount,
    constant uint& zCount,
    constant uint& yCount,
    uint index
) {
    uint outputCount = xCount * yCount;
    if (index >= outputCount) {
        return;
    }

    uint xIndex = index / yCount;
    uint yIndex = index % yCount;
    float total = 0.0f;

    for (uint zIndex = 0; zIndex < zCount; zIndex++) {
        uint conditionalIndex = (xIndex * zCount + zIndex) * yCount + yIndex;
        total += Storage::load(conditional, conditionalIndex) * Storage::load(marginal, zIndex);
    }

    Storage::store(out, index, total);
}

template <typename Storage, typename Scalar>
static inline void frontdoor_adjustment_kernel(
    device const Scalar* mediator,
    device const Scalar* outcome,
    device const Scalar* marginal,
    device Scalar* out,
    constant uint& xCount,
    constant uint& mCount,
    constant uint& yCount,
    uint index
) {
    uint outputCount = xCount * yCount;
    if (index >= outputCount) {
        return;
    }

    uint xIndex = index / yCount;
    uint yIndex = index % yCount;
    float total = 0.0f;

    for (uint mIndex = 0; mIndex < mCount; mIndex++) {
        float mediatorValue = Storage::load(mediator, xIndex * mCount + mIndex);
        float innerSum = 0.0f;

        for (uint xPrimeIndex = 0; xPrimeIndex < xCount; xPrimeIndex++) {
            uint outcomeIndex = (xPrimeIndex * mCount + mIndex) * yCount + yIndex;
            innerSum += Storage::load(outcome, outcomeIndex) * Storage::load(marginal, xPrimeIndex);
        }

        total += mediatorValue * innerSum;
    }

    Storage::store(out, index, total);
}

template <typename Storage, typename Scalar>
static inline void do_intervene_kernel(
    device const Scalar* adjacency,
    device const int* intervened,
    device Scalar* out,
    constant uint& nodeCount,
    constant uint& intervenedCount,
    uint index
) {
    uint matrixCount = nodeCount * nodeCount;
    if (index >= matrixCount) {
        return;
    }

    uint targetNode = index % nodeCount;
    bool removeIncoming = false;

    for (uint nodeIndex = 0; nodeIndex < intervenedCount; nodeIndex++) {
        int candidate = intervened[nodeIndex];
        if (candidate >= 0 && uint(candidate) == targetNode) {
            removeIncoming = true;
        }
    }

    Storage::store(out, index, removeIncoming ? 0.0f : Storage::load(adjacency, index));
}

template <typename Storage, typename Scalar>
static inline void cate_kernel(
    device const Scalar* treated,
    device const Scalar* control,
    device Scalar* out,
    constant uint& count,
    uint index
) {
    if (index < count) {
        Storage::store(out, index, Storage::load(treated, index) - Storage::load(control, index));
    }
}

template <typename Storage, typename Scalar>
static inline void counterfactual_kernel(
    device const Scalar* observedY,
    device const Scalar* observedX,
    device const Scalar* counterfactualX,
    device const Scalar* slope,
    device Scalar* out,
    constant uint& count,
    uint index
) {
    if (index >= count) {
        return;
    }

    float slopeValue = Storage::load(slope, 0);
    float value = Storage::load(observedY, index) +
        slopeValue * (Storage::load(counterfactualX, index) - Storage::load(observedX, index));
    Storage::store(out, index, value);
}

static inline void causal_reduce_sum(threadgroup float* values, uint threadIndex) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = causalThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            values[threadIndex] += values[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <typename Storage, typename Scalar>
static inline void iv_estimate_partial_kernel(
    device const Scalar* instrument,
    device const Scalar* treatment,
    device const Scalar* outcome,
    device float* scratch,
    threadgroup float* sumZ,
    threadgroup float* sumX,
    threadgroup float* sumY,
    threadgroup float* sumZY,
    threadgroup float* sumZX,
    constant uint& count,
    uint groupIndex,
    uint threadIndex
) {
    uint valueIndex = groupIndex * causalThreadCount + threadIndex;
    float zValue = 0.0f;
    float xValue = 0.0f;
    float yValue = 0.0f;

    if (valueIndex < count) {
        zValue = Storage::load(instrument, valueIndex);
        xValue = Storage::load(treatment, valueIndex);
        yValue = Storage::load(outcome, valueIndex);
    }

    sumZ[threadIndex] = zValue;
    sumX[threadIndex] = xValue;
    sumY[threadIndex] = yValue;
    sumZY[threadIndex] = zValue * yValue;
    sumZX[threadIndex] = zValue * xValue;
    causal_reduce_sum(sumZ, threadIndex);
    causal_reduce_sum(sumX, threadIndex);
    causal_reduce_sum(sumY, threadIndex);
    causal_reduce_sum(sumZY, threadIndex);
    causal_reduce_sum(sumZX, threadIndex);

    if (threadIndex == 0) {
        uint offset = groupIndex * 5;
        scratch[offset] = sumZ[0];
        scratch[offset + 1] = sumX[0];
        scratch[offset + 2] = sumY[0];
        scratch[offset + 3] = sumZY[0];
        scratch[offset + 4] = sumZX[0];
    }
}

template <typename Storage, typename Scalar>
static inline void iv_estimate_finalize_kernel(
    device const float* scratch,
    device Scalar* out,
    threadgroup float* sumZ,
    threadgroup float* sumX,
    threadgroup float* sumY,
    threadgroup float* sumZY,
    threadgroup float* sumZX,
    constant uint& count,
    constant uint& partialCount,
    uint threadIndex
) {
    float localZ = 0.0f;
    float localX = 0.0f;
    float localY = 0.0f;
    float localZY = 0.0f;
    float localZX = 0.0f;

    for (uint partialIndex = threadIndex; partialIndex < partialCount; partialIndex += causalThreadCount) {
        uint offset = partialIndex * 5;
        localZ += scratch[offset];
        localX += scratch[offset + 1];
        localY += scratch[offset + 2];
        localZY += scratch[offset + 3];
        localZX += scratch[offset + 4];
    }

    sumZ[threadIndex] = localZ;
    sumX[threadIndex] = localX;
    sumY[threadIndex] = localY;
    sumZY[threadIndex] = localZY;
    sumZX[threadIndex] = localZX;
    causal_reduce_sum(sumZ, threadIndex);
    causal_reduce_sum(sumX, threadIndex);
    causal_reduce_sum(sumY, threadIndex);
    causal_reduce_sum(sumZY, threadIndex);
    causal_reduce_sum(sumZX, threadIndex);

    if (threadIndex == 0) {
        float denominator = sumZX[0] - (sumZ[0] * sumX[0]) / float(count);
        float numerator = sumZY[0] - (sumZ[0] * sumY[0]) / float(count);
        Storage::store(out, 0, fabs(denominator) < 1.0e-12f ? 0.0f : numerator / denominator);
    }
}

template <typename Storage, typename Scalar>
static inline void dag_markov_factorization_partial_kernel(
    device const Scalar* conditionals,
    device float* scratch,
    threadgroup float* products,
    constant uint& count,
    uint groupIndex,
    uint threadIndex
) {
    uint valueIndex = groupIndex * causalThreadCount + threadIndex;
    products[threadIndex] = valueIndex < count ?
        causal_safe_positive(Storage::load(conditionals, valueIndex)) : 1.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = causalThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            products[threadIndex] *= products[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        scratch[groupIndex] = products[0];
    }
}

template <typename Storage, typename Scalar>
static inline void dag_markov_factorization_finalize_kernel(
    device const float* scratch,
    device Scalar* out,
    threadgroup float* products,
    constant uint& partialCount,
    uint threadIndex
) {
    float localProduct = 1.0f;

    for (uint partialIndex = threadIndex; partialIndex < partialCount; partialIndex += causalThreadCount) {
        localProduct *= scratch[partialIndex];
    }

    products[threadIndex] = localProduct;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = causalThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            products[threadIndex] *= products[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        Storage::store(out, 0, products[0]);
    }
}

#define BACKDOOR_KERNEL(name, storage, scalar) \
kernel void name(device const scalar* conditional [[buffer(0)]], device const scalar* marginal [[buffer(1)]], device scalar* out [[buffer(2)]], constant uint& xCount [[buffer(3)]], constant uint& zCount [[buffer(4)]], constant uint& yCount [[buffer(5)]], uint index [[thread_position_in_grid]]) { \
    backdoor_adjustment_kernel<storage, scalar>(conditional, marginal, out, xCount, zCount, yCount, index); \
}

#define FRONTDOOR_KERNEL(name, storage, scalar) \
kernel void name(device const scalar* mediator [[buffer(0)]], device const scalar* outcome [[buffer(1)]], device const scalar* marginal [[buffer(2)]], device scalar* out [[buffer(3)]], constant uint& xCount [[buffer(4)]], constant uint& mCount [[buffer(5)]], constant uint& yCount [[buffer(6)]], uint index [[thread_position_in_grid]]) { \
    frontdoor_adjustment_kernel<storage, scalar>(mediator, outcome, marginal, out, xCount, mCount, yCount, index); \
}

#define DO_INTERVENE_KERNEL(name, storage, scalar) \
kernel void name(device const scalar* adjacency [[buffer(0)]], device const int* intervened [[buffer(1)]], device scalar* out [[buffer(2)]], constant uint& nodeCount [[buffer(3)]], constant uint& intervenedCount [[buffer(4)]], uint index [[thread_position_in_grid]]) { \
    do_intervene_kernel<storage, scalar>(adjacency, intervened, out, nodeCount, intervenedCount, index); \
}

#define CATE_KERNEL(name, storage, scalar) \
kernel void name(device const scalar* treated [[buffer(0)]], device const scalar* control [[buffer(1)]], device scalar* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint index [[thread_position_in_grid]]) { \
    cate_kernel<storage, scalar>(treated, control, out, count, index); \
}

#define COUNTERFACTUAL_KERNEL(name, storage, scalar) \
kernel void name(device const scalar* observedY [[buffer(0)]], device const scalar* observedX [[buffer(1)]], device const scalar* counterfactualX [[buffer(2)]], device const scalar* slope [[buffer(3)]], device scalar* out [[buffer(4)]], constant uint& count [[buffer(5)]], uint index [[thread_position_in_grid]]) { \
    counterfactual_kernel<storage, scalar>(observedY, observedX, counterfactualX, slope, out, count, index); \
}

#define IV_KERNELS(prefix, storage, scalar) \
kernel void prefix##_partial(device const scalar* instrument [[buffer(0)]], device const scalar* treatment [[buffer(1)]], device const scalar* outcome [[buffer(2)]], device float* scratch [[buffer(3)]], constant uint& count [[buffer(4)]], uint groupIndex [[threadgroup_position_in_grid]], uint threadIndex [[thread_position_in_threadgroup]]) { \
    threadgroup float sumZ[256]; threadgroup float sumX[256]; threadgroup float sumY[256]; threadgroup float sumZY[256]; threadgroup float sumZX[256]; \
    iv_estimate_partial_kernel<storage, scalar>(instrument, treatment, outcome, scratch, sumZ, sumX, sumY, sumZY, sumZX, count, groupIndex, threadIndex); \
} \
kernel void prefix##_finalize(device const float* scratch [[buffer(0)]], device scalar* out [[buffer(1)]], constant uint& count [[buffer(2)]], constant uint& partialCount [[buffer(3)]], uint threadIndex [[thread_position_in_threadgroup]]) { \
    threadgroup float sumZ[256]; threadgroup float sumX[256]; threadgroup float sumY[256]; threadgroup float sumZY[256]; threadgroup float sumZX[256]; \
    iv_estimate_finalize_kernel<storage, scalar>(scratch, out, sumZ, sumX, sumY, sumZY, sumZX, count, partialCount, threadIndex); \
}

#define DAG_KERNELS(prefix, storage, scalar) \
kernel void prefix##_partial(device const scalar* conditionals [[buffer(0)]], device float* scratch [[buffer(1)]], constant uint& count [[buffer(2)]], uint groupIndex [[threadgroup_position_in_grid]], uint threadIndex [[thread_position_in_threadgroup]]) { \
    threadgroup float products[256]; \
    dag_markov_factorization_partial_kernel<storage, scalar>(conditionals, scratch, products, count, groupIndex, threadIndex); \
} \
kernel void prefix##_finalize(device const float* scratch [[buffer(0)]], device scalar* out [[buffer(1)]], constant uint& partialCount [[buffer(2)]], uint threadIndex [[thread_position_in_threadgroup]]) { \
    threadgroup float products[256]; \
    dag_markov_factorization_finalize_kernel<storage, scalar>(scratch, out, products, partialCount, threadIndex); \
}

BACKDOOR_KERNEL(backdoor_adjustment_float32, Float32CausalStorage, float)
BACKDOOR_KERNEL(backdoor_adjustment_float16, Float16CausalStorage, half)
BACKDOOR_KERNEL(backdoor_adjustment_bfloat16, BFloat16CausalStorage, ushort)
FRONTDOOR_KERNEL(frontdoor_adjustment_float32, Float32CausalStorage, float)
FRONTDOOR_KERNEL(frontdoor_adjustment_float16, Float16CausalStorage, half)
FRONTDOOR_KERNEL(frontdoor_adjustment_bfloat16, BFloat16CausalStorage, ushort)
DO_INTERVENE_KERNEL(do_intervene_float32, Float32CausalStorage, float)
DO_INTERVENE_KERNEL(do_intervene_float16, Float16CausalStorage, half)
DO_INTERVENE_KERNEL(do_intervene_bfloat16, BFloat16CausalStorage, ushort)
CATE_KERNEL(cate_float32, Float32CausalStorage, float)
CATE_KERNEL(cate_float16, Float16CausalStorage, half)
CATE_KERNEL(cate_bfloat16, BFloat16CausalStorage, ushort)
COUNTERFACTUAL_KERNEL(counterfactual_float32, Float32CausalStorage, float)
COUNTERFACTUAL_KERNEL(counterfactual_float16, Float16CausalStorage, half)
COUNTERFACTUAL_KERNEL(counterfactual_bfloat16, BFloat16CausalStorage, ushort)
IV_KERNELS(iv_estimate_float32, Float32CausalStorage, float)
IV_KERNELS(iv_estimate_float16, Float16CausalStorage, half)
IV_KERNELS(iv_estimate_bfloat16, BFloat16CausalStorage, ushort)
DAG_KERNELS(dag_markov_factorization_float32, Float32CausalStorage, float)
DAG_KERNELS(dag_markov_factorization_float16, Float16CausalStorage, half)
DAG_KERNELS(dag_markov_factorization_bfloat16, BFloat16CausalStorage, ushort)
