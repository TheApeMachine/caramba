#include <metal_stdlib>

using namespace metal;

constant uint activeThreadCount = 256;

static inline float active_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort active_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline float active_safe_positive(float value) {
    return max(value, 1.0e-12f);
}

struct Float32ActiveStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16ActiveStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16ActiveStorage {
    static float load(device const ushort* values, uint index) {
        return active_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = active_float_to_bf16(value);
    }
};

template <typename Storage, typename Scalar>
static inline void free_energy_partial(
    device const Scalar* likelihood,
    device const Scalar* posterior,
    device const Scalar* prior,
    device float* scratch,
    threadgroup float* reduction,
    constant uint& count,
    uint groupIndex,
    uint threadIndex
) {
    uint valueIndex = groupIndex * activeThreadCount + threadIndex;
    float localValue = 0.0f;

    if (valueIndex < count) {
        float posteriorValue = Storage::load(posterior, valueIndex);
        float safeLikelihood = active_safe_positive(Storage::load(likelihood, valueIndex));
        float safePosterior = active_safe_positive(posteriorValue);
        float safePrior = active_safe_positive(Storage::load(prior, valueIndex));
        localValue = posteriorValue * (-log(safeLikelihood) + log(safePosterior) - log(safePrior));
    }

    reduction[threadIndex] = localValue;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = activeThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        scratch[groupIndex] = reduction[0];
    }
}

template <typename Storage, typename Scalar>
static inline void expected_free_energy_partial(
    device const Scalar* predictedObs,
    device const Scalar* preferredObs,
    device const Scalar* predictedState,
    device float* scratch,
    threadgroup float* reduction,
    constant uint& obsCount,
    constant uint& stateCount,
    constant uint& obsPartialCount,
    uint groupIndex,
    uint threadIndex
) {
    bool obsGroup = groupIndex < obsPartialCount;
    uint localGroup = obsGroup ? groupIndex : groupIndex - obsPartialCount;
    uint valueIndex = localGroup * activeThreadCount + threadIndex;
    float localValue = 0.0f;

    if (obsGroup && valueIndex < obsCount) {
        float predicted = Storage::load(predictedObs, valueIndex);
        float safePredicted = active_safe_positive(predicted);
        float safePreferred = active_safe_positive(Storage::load(preferredObs, valueIndex));
        localValue = predicted * (log(safePredicted) - log(safePreferred));
    }

    if (!obsGroup && valueIndex < stateCount) {
        float state = Storage::load(predictedState, valueIndex);
        localValue = -state * log(active_safe_positive(state));
    }

    reduction[threadIndex] = localValue;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = activeThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        scratch[groupIndex] = reduction[0];
    }
}

template <typename Storage, typename Scalar>
static inline void active_scalar_finalize(
    device const float* scratch,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& partialCount,
    uint threadIndex
) {
    float localValue = 0.0f;

    for (uint index = threadIndex; index < partialCount; index += activeThreadCount) {
        localValue += scratch[index];
    }

    reduction[threadIndex] = localValue;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = activeThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        Storage::store(out, 0, reduction[0]);
    }
}

template <typename Storage, typename Scalar>
static inline void belief_update_partial(
    device const Scalar* likelihood,
    device const Scalar* prior,
    device float* scratch,
    threadgroup float* reduction,
    constant uint& count,
    uint groupIndex,
    uint threadIndex
) {
    uint valueIndex = groupIndex * activeThreadCount + threadIndex;
    float localValue = 0.0f;

    if (valueIndex < count) {
        localValue = Storage::load(likelihood, valueIndex) * Storage::load(prior, valueIndex);
    }

    reduction[threadIndex] = localValue;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = activeThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        scratch[groupIndex] = reduction[0];
    }
}

template <typename Storage, typename Scalar>
static inline void belief_update_normalize(
    device const Scalar* likelihood,
    device const Scalar* prior,
    device const float* scratch,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& count,
    constant uint& partialCount,
    uint threadIndex
) {
    float localSum = 0.0f;

    for (uint index = threadIndex; index < partialCount; index += activeThreadCount) {
        localSum += scratch[index];
    }

    reduction[threadIndex] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = activeThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float total = reduction[0];

    for (uint index = threadIndex; index < count; index += activeThreadCount) {
        float value = Storage::load(likelihood, index) * Storage::load(prior, index);
        if (total != 0.0f) {
            value /= total;
        }

        Storage::store(out, index, value);
    }
}

template <typename Storage, typename Scalar>
static inline void precision_weight_kernel(
    device const Scalar* errors,
    device const Scalar* precision,
    device Scalar* out,
    constant uint& count,
    uint index
) {
    if (index >= count) {
        return;
    }

    Storage::store(out, index, Storage::load(errors, index) * Storage::load(precision, index));
}

#define ACTIVE_SCALAR_PARTIAL_KERNEL(name, body, storage, scalar) \
kernel void name( \
    device const scalar* first [[buffer(0)]], \
    device const scalar* second [[buffer(1)]], \
    device const scalar* third [[buffer(2)]], \
    device float* scratch [[buffer(3)]], \
    constant uint& count [[buffer(4)]], \
    uint groupIndex [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    body<storage, scalar>(first, second, third, scratch, reduction, count, groupIndex, threadIndex); \
}

#define EXPECTED_FREE_ENERGY_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* predictedObs [[buffer(0)]], \
    device const scalar* preferredObs [[buffer(1)]], \
    device const scalar* predictedState [[buffer(2)]], \
    device float* scratch [[buffer(3)]], \
    constant uint& obsCount [[buffer(4)]], \
    constant uint& stateCount [[buffer(5)]], \
    constant uint& obsPartialCount [[buffer(6)]], \
    uint groupIndex [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    expected_free_energy_partial<storage, scalar>( \
        predictedObs, preferredObs, predictedState, scratch, reduction, \
        obsCount, stateCount, obsPartialCount, groupIndex, threadIndex \
    ); \
}

#define ACTIVE_FINALIZE_KERNEL(name, storage, scalar) \
kernel void name( \
    device const float* scratch [[buffer(0)]], \
    device scalar* out [[buffer(1)]], \
    constant uint& partialCount [[buffer(2)]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    active_scalar_finalize<storage, scalar>(scratch, out, reduction, partialCount, threadIndex); \
}

#define BELIEF_PARTIAL_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* likelihood [[buffer(0)]], \
    device const scalar* prior [[buffer(1)]], \
    device float* scratch [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint groupIndex [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    belief_update_partial<storage, scalar>( \
        likelihood, prior, scratch, reduction, count, groupIndex, threadIndex \
    ); \
}

#define BELIEF_NORMALIZE_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* likelihood [[buffer(0)]], \
    device const scalar* prior [[buffer(1)]], \
    device const float* scratch [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& count [[buffer(4)]], \
    constant uint& partialCount [[buffer(5)]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    belief_update_normalize<storage, scalar>( \
        likelihood, prior, scratch, out, reduction, count, partialCount, threadIndex \
    ); \
}

#define PRECISION_WEIGHT_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* errors [[buffer(0)]], \
    device const scalar* precision [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint index [[thread_position_in_grid]] \
) { \
    precision_weight_kernel<storage, scalar>(errors, precision, out, count, index); \
}

ACTIVE_SCALAR_PARTIAL_KERNEL(free_energy_float32_partial, free_energy_partial, Float32ActiveStorage, float)
EXPECTED_FREE_ENERGY_KERNEL(expected_free_energy_float32_partial, Float32ActiveStorage, float)
ACTIVE_FINALIZE_KERNEL(active_scalar_finalize_float32_value, Float32ActiveStorage, float)
BELIEF_PARTIAL_KERNEL(belief_update_float32_partial, Float32ActiveStorage, float)
BELIEF_NORMALIZE_KERNEL(belief_update_float32_normalize, Float32ActiveStorage, float)
PRECISION_WEIGHT_KERNEL(precision_weight_float32, Float32ActiveStorage, float)

ACTIVE_SCALAR_PARTIAL_KERNEL(free_energy_float16_partial, free_energy_partial, Float16ActiveStorage, half)
EXPECTED_FREE_ENERGY_KERNEL(expected_free_energy_float16_partial, Float16ActiveStorage, half)
ACTIVE_FINALIZE_KERNEL(active_scalar_finalize_float16_value, Float16ActiveStorage, half)
BELIEF_PARTIAL_KERNEL(belief_update_float16_partial, Float16ActiveStorage, half)
BELIEF_NORMALIZE_KERNEL(belief_update_float16_normalize, Float16ActiveStorage, half)
PRECISION_WEIGHT_KERNEL(precision_weight_float16, Float16ActiveStorage, half)

ACTIVE_SCALAR_PARTIAL_KERNEL(free_energy_bfloat16_partial, free_energy_partial, BFloat16ActiveStorage, ushort)
EXPECTED_FREE_ENERGY_KERNEL(expected_free_energy_bfloat16_partial, BFloat16ActiveStorage, ushort)
ACTIVE_FINALIZE_KERNEL(active_scalar_finalize_bfloat16_value, BFloat16ActiveStorage, ushort)
BELIEF_PARTIAL_KERNEL(belief_update_bfloat16_partial, BFloat16ActiveStorage, ushort)
BELIEF_NORMALIZE_KERNEL(belief_update_bfloat16_normalize, BFloat16ActiveStorage, ushort)
PRECISION_WEIGHT_KERNEL(precision_weight_bfloat16, BFloat16ActiveStorage, ushort)
