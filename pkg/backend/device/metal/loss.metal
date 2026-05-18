#include <metal_stdlib>

using namespace metal;

constant uint lossThreadCount = 256;

static inline float loss_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort loss_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32LossStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16LossStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16LossStorage {
    static float load(device const ushort* values, uint index) {
        return loss_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = loss_float_to_bf16(value);
    }
};

static inline void set_loss_error(device atomic_uint* errorFlag) {
    atomic_store_explicit(errorFlag, 1u, memory_order_relaxed);
}

static inline float loss_safe_probability(float value) {
    return clamp(value, 1.0e-7f, 1.0f - 1.0e-7f);
}

static inline float loss_safe_positive(float value) {
    return max(value, 1.0e-12f);
}

struct MSELossOp {
    float operator()(float prediction, float target) const {
        float delta = prediction - target;
        return delta * delta;
    }
};

struct MAELossOp {
    float operator()(float prediction, float target) const {
        return abs(prediction - target);
    }
};

struct HuberLossOp {
    float operator()(float prediction, float target) const {
        float delta = prediction - target;
        float magnitude = abs(delta);

        if (magnitude <= 1.0f) {
            return 0.5f * delta * delta;
        }

        return magnitude - 0.5f;
    }
};

struct BinaryCrossEntropyLossOp {
    float operator()(float prediction, float target) const {
        float safePrediction = loss_safe_probability(prediction);
        return -target * log(safePrediction) -
            (1.0f - target) * log(1.0f - safePrediction);
    }
};

struct KLDivergenceLossOp {
    float operator()(float prediction, float target) const {
        float safePrediction = loss_safe_positive(prediction);
        float safeTarget = loss_safe_positive(target);
        return safeTarget * log(safeTarget / safePrediction);
    }
};

template <typename Storage, typename Scalar, typename LossOp>
static inline void pair_loss_partial(
    device const Scalar* predictions,
    device const Scalar* targets,
    device float* scratch,
    threadgroup float* reduction,
    constant uint& count,
    uint groupIndex,
    uint threadIndex,
    LossOp op
) {
    uint valueIndex = groupIndex * lossThreadCount + threadIndex;
    float localValue = 0.0f;

    if (valueIndex < count) {
        localValue = op(
            Storage::load(predictions, valueIndex),
            Storage::load(targets, valueIndex)
        );
    }

    reduction[threadIndex] = localValue;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lossThreadCount / 2; stride > 0; stride >>= 1) {
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
static inline void loss_finalize(
    device const float* scratch,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& partialCount,
    constant uint& denominator,
    uint threadIndex
) {
    float localValue = 0.0f;

    for (uint index = threadIndex; index < partialCount; index += lossThreadCount) {
        localValue += scratch[index];
    }

    reduction[threadIndex] = localValue;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lossThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        Storage::store(out, 0, reduction[0] / float(denominator));
    }
}

template <typename Storage, typename Scalar>
static inline void cross_entropy_loss_partial(
    device const Scalar* logits,
    device const int* targets,
    device float* scratch,
    device atomic_uint* errorFlag,
    threadgroup float* reduction,
    constant uint& classes,
    uint rowIndex,
    uint threadIndex
) {
    int targetID = targets[rowIndex];
    bool targetOK = targetID >= 0 && uint(targetID) < classes;
    uint rowOffset = rowIndex * classes;
    float localMax = -3.4028234663852886e38f;

    if (!targetOK && threadIndex == 0) {
        set_loss_error(errorFlag);
    }

    for (uint col = threadIndex; targetOK && col < classes; col += lossThreadCount) {
        localMax = max(localMax, Storage::load(logits, rowOffset + col));
    }

    reduction[threadIndex] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lossThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] = max(reduction[threadIndex], reduction[threadIndex + stride]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float maximum = reduction[0];
    float localSum = 0.0f;

    for (uint col = threadIndex; targetOK && col < classes; col += lossThreadCount) {
        localSum += exp(Storage::load(logits, rowOffset + col) - maximum);
    }

    reduction[threadIndex] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lossThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        float targetLogit = targetOK ? Storage::load(logits, rowOffset + uint(targetID)) : 0.0f;
        scratch[rowIndex] = targetOK ? -(targetLogit - maximum - log(reduction[0])) : 0.0f;
    }
}

#define PAIR_LOSS_KERNEL(name, storage, scalar, op) \
kernel void name##_partial( \
    device const scalar* predictions [[buffer(0)]], \
    device const scalar* targets [[buffer(1)]], \
    device float* scratch [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint groupIndex [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    pair_loss_partial<storage, scalar>( \
        predictions, targets, scratch, reduction, count, groupIndex, threadIndex, op{} \
    ); \
}

#define LOSS_FINALIZE_KERNEL(name, storage, scalar) \
kernel void name( \
    device const float* scratch [[buffer(0)]], \
    device scalar* out [[buffer(1)]], \
    constant uint& partialCount [[buffer(2)]], \
    constant uint& denominator [[buffer(3)]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    loss_finalize<storage, scalar>(scratch, out, reduction, partialCount, denominator, threadIndex); \
}

#define CROSS_ENTROPY_KERNEL(name, storage, scalar) \
kernel void name##_partial( \
    device const scalar* logits [[buffer(0)]], \
    device const int* targets [[buffer(1)]], \
    device float* scratch [[buffer(2)]], \
    device atomic_uint* errorFlag [[buffer(3)]], \
    constant uint& batch [[buffer(4)]], \
    constant uint& classes [[buffer(5)]], \
    uint rowIndex [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    if (rowIndex >= batch) { \
        return; \
    } \
    threadgroup float reduction[256]; \
    cross_entropy_loss_partial<storage, scalar>( \
        logits, targets, scratch, errorFlag, reduction, classes, rowIndex, threadIndex \
    ); \
}

PAIR_LOSS_KERNEL(mse_loss_float32, Float32LossStorage, float, MSELossOp)
PAIR_LOSS_KERNEL(mae_loss_float32, Float32LossStorage, float, MAELossOp)
PAIR_LOSS_KERNEL(huber_loss_float32, Float32LossStorage, float, HuberLossOp)
PAIR_LOSS_KERNEL(binary_cross_entropy_float32, Float32LossStorage, float, BinaryCrossEntropyLossOp)
PAIR_LOSS_KERNEL(kl_divergence_float32, Float32LossStorage, float, KLDivergenceLossOp)
LOSS_FINALIZE_KERNEL(loss_finalize_float32, Float32LossStorage, float)
CROSS_ENTROPY_KERNEL(cross_entropy_float32, Float32LossStorage, float)

PAIR_LOSS_KERNEL(mse_loss_float16, Float16LossStorage, half, MSELossOp)
PAIR_LOSS_KERNEL(mae_loss_float16, Float16LossStorage, half, MAELossOp)
PAIR_LOSS_KERNEL(huber_loss_float16, Float16LossStorage, half, HuberLossOp)
PAIR_LOSS_KERNEL(binary_cross_entropy_float16, Float16LossStorage, half, BinaryCrossEntropyLossOp)
PAIR_LOSS_KERNEL(kl_divergence_float16, Float16LossStorage, half, KLDivergenceLossOp)
LOSS_FINALIZE_KERNEL(loss_finalize_float16, Float16LossStorage, half)
CROSS_ENTROPY_KERNEL(cross_entropy_float16, Float16LossStorage, half)

PAIR_LOSS_KERNEL(mse_loss_bfloat16, BFloat16LossStorage, ushort, MSELossOp)
PAIR_LOSS_KERNEL(mae_loss_bfloat16, BFloat16LossStorage, ushort, MAELossOp)
PAIR_LOSS_KERNEL(huber_loss_bfloat16, BFloat16LossStorage, ushort, HuberLossOp)
PAIR_LOSS_KERNEL(binary_cross_entropy_bfloat16, BFloat16LossStorage, ushort, BinaryCrossEntropyLossOp)
PAIR_LOSS_KERNEL(kl_divergence_bfloat16, BFloat16LossStorage, ushort, KLDivergenceLossOp)
LOSS_FINALIZE_KERNEL(loss_finalize_bfloat16, BFloat16LossStorage, ushort)
CROSS_ENTROPY_KERNEL(cross_entropy_bfloat16, BFloat16LossStorage, ushort)
