#include <metal_stdlib>

using namespace metal;

constant float adamLearningRate = 1.0e-4f;
constant float adamBeta1 = 0.9f;
constant float adamBeta2 = 0.999f;
constant float adamEpsilon = 1.0e-8f;
constant float adamWBeta1 = 0.9f;
constant float adamWBeta2 = 0.999f;
constant float adamWLearningRate = 1.0e-4f;
constant float adamWEpsilon = 1.0e-8f;
constant float adamWDecay = 1.0e-2f;
constant float adamaxLearningRate = 2.0e-3f;
constant float adamaxBeta1 = 0.9f;
constant float adamaxBeta2 = 0.999f;
constant float adamaxEpsilon = 1.0e-8f;
constant float adagradLearningRate = 1.0e-2f;
constant float adagradEpsilon = 1.0e-10f;
constant float rmspropLearningRate = 1.0e-3f;
constant float rmspropDecay = 0.99f;
constant float rmspropEpsilon = 1.0e-8f;
constant float lionLearningRate = 1.0e-4f;
constant float lionBeta1 = 0.9f;
constant float lionBeta2 = 0.99f;
constant float sgdLearningRate = 1.0e-2f;
constant float sgdMomentum = 0.9f;
constant float larsLearningRate = 1.0e-2f;
constant float larsMomentum = 0.9f;
constant float larsWeightDecay = 1.0e-4f;
constant float larsTrustCoeff = 1.0e-3f;
constant float larsEpsilon = 1.0e-8f;
constant float hebbianLearningRate = 1.0e-3f;
constant float hebbianDecay = 1.0e-4f;

static inline float optimizer_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort optimizer_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32OptimizerStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16OptimizerStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16OptimizerStorage {
    static float load(device const ushort* values, uint index) {
        return optimizer_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = optimizer_float_to_bf16(value);
    }
};

static inline float optimizer_sign(float value) {
    if (value > 0.0f) {
        return 1.0f;
    }

    if (value < 0.0f) {
        return -1.0f;
    }

    return 0.0f;
}

template <typename Storage, typename Scalar>
static inline void adam_step_kernel(
    device const Scalar* params,
    device const Scalar* gradients,
    device float* first,
    device float* second,
    device Scalar* out,
    uint count,
    uint index
) {
    if (index >= count) {
        return;
    }

    float param = Storage::load(params, index);
    float grad = Storage::load(gradients, index);
    first[index] = adamBeta1 * first[index] + (1.0f - adamBeta1) * grad;
    second[index] = adamBeta2 * second[index] + (1.0f - adamBeta2) * grad * grad;
    float correctedFirst = first[index] / (1.0f - adamBeta1);
    float correctedSecond = second[index] / (1.0f - adamBeta2);
    float denominator = sqrt(correctedSecond) + adamEpsilon;
    Storage::store(out, index, param - adamLearningRate * correctedFirst / denominator);
}

template <typename Storage, typename Scalar>
static inline void adamw_step_kernel(
    device const Scalar* params,
    device const Scalar* gradients,
    device float* first,
    device float* second,
    device Scalar* out,
    uint count,
    uint index
) {
    if (index >= count) {
        return;
    }

    float param = Storage::load(params, index);
    float grad = Storage::load(gradients, index);
    first[index] = adamWBeta1 * first[index] + (1.0f - adamWBeta1) * grad;
    second[index] = adamWBeta2 * second[index] + (1.0f - adamWBeta2) * grad * grad;
    float correctedFirst = first[index] / (1.0f - adamWBeta1);
    float correctedSecond = second[index] / (1.0f - adamWBeta2);
    float denominator = sqrt(correctedSecond) + adamWEpsilon;
    float gradStep = adamWLearningRate * correctedFirst / denominator;
    float decayStep = adamWLearningRate * adamWDecay * param;
    Storage::store(out, index, param - gradStep - decayStep);
}

template <typename Storage, typename Scalar>
static inline void adamax_step_kernel(
    device const Scalar* params,
    device const Scalar* gradients,
    device float* first,
    device float* infinity,
    device Scalar* out,
    uint count,
    uint index
) {
    if (index >= count) {
        return;
    }

    float param = Storage::load(params, index);
    float grad = Storage::load(gradients, index);
    first[index] = adamaxBeta1 * first[index] + (1.0f - adamaxBeta1) * grad;
    infinity[index] = max(adamaxBeta2 * infinity[index], abs(grad));
    float correctedFirst = first[index] / (1.0f - adamaxBeta1);
    Storage::store(
        out, index,
        param - adamaxLearningRate * correctedFirst / (infinity[index] + adamaxEpsilon)
    );
}

template <typename Storage, typename Scalar>
static inline void optimizer3_step_kernel(
    device const Scalar* params,
    device const Scalar* gradients,
    device float* state,
    device Scalar* out,
    uint count,
    uint operation,
    uint index
) {
    if (index >= count) {
        return;
    }

    float param = Storage::load(params, index);
    float grad = Storage::load(gradients, index);

    if (operation == 3u) {
        state[index] += grad * grad;
        Storage::store(out, index, param - adagradLearningRate * grad / (sqrt(state[index]) + adagradEpsilon));
        return;
    }

    if (operation == 4u) {
        state[index] = rmspropDecay * state[index] + (1.0f - rmspropDecay) * grad * grad;
        Storage::store(out, index, param - rmspropLearningRate * grad / (sqrt(state[index]) + rmspropEpsilon));
        return;
    }

    if (operation == 5u) {
        float update = lionBeta1 * state[index] + (1.0f - lionBeta1) * grad;
        Storage::store(out, index, param - lionLearningRate * optimizer_sign(update));
        state[index] = lionBeta2 * state[index] + (1.0f - lionBeta2) * grad;
        return;
    }

    state[index] = sgdMomentum * state[index] + grad;
    Storage::store(out, index, param - sgdLearningRate * state[index]);
}

template <typename Storage, typename Scalar>
static inline void lbfgs_step_kernel(
    device const Scalar* params,
    device const Scalar* gradients,
    device Scalar* out,
    uint count,
    uint index
) {
    if (index >= count) {
        return;
    }

    Storage::store(
        out, index,
        Storage::load(params, index) - Storage::load(gradients, index)
    );
}

template <typename Storage, typename Scalar>
static inline void hebbian_step_kernel(
    device const Scalar* weights,
    device const Scalar* post,
    device const Scalar* pre,
    device Scalar* out,
    uint postCount,
    uint preCount,
    uint index
) {
    uint count = postCount * preCount;

    if (index >= count) {
        return;
    }

    uint postIndex = index / preCount;
    uint preIndex = index - postIndex * preCount;
    float updated = Storage::load(weights, index) * (1.0f - hebbianDecay) +
        hebbianLearningRate * Storage::load(post, postIndex) * Storage::load(pre, preIndex);
    Storage::store(out, index, updated);
}

template <typename Storage, typename Scalar>
static inline void lars_norms_kernel(
    device const Scalar* params,
    device const Scalar* gradients,
    device float* scratch,
    threadgroup float* paramSums,
    threadgroup float* gradSums,
    uint count,
    uint localIndex,
    uint groupIndex
) {
    uint index = groupIndex * 256u + localIndex;
    float param = 0.0f;
    float grad = 0.0f;

    if (index < count) {
        param = Storage::load(params, index);
        grad = Storage::load(gradients, index);
    }

    paramSums[localIndex] = param * param;
    gradSums[localIndex] = grad * grad;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (localIndex < stride) {
            paramSums[localIndex] += paramSums[localIndex + stride];
            gradSums[localIndex] += gradSums[localIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (localIndex == 0u) {
        scratch[groupIndex * 2u] = paramSums[0];
        scratch[groupIndex * 2u + 1u] = gradSums[0];
    }
}

template <typename Storage, typename Scalar>
static inline void lars_step_kernel(
    device const Scalar* params,
    device const Scalar* gradients,
    device float* momentum,
    device const float* scratch,
    device Scalar* out,
    uint count,
    uint groupCount,
    uint index
) {
    if (index >= count) {
        return;
    }

    float paramSum = 0.0f;
    float gradSum = 0.0f;

    for (uint groupIndex = 0; groupIndex < groupCount; groupIndex++) {
        paramSum += scratch[groupIndex * 2u];
        gradSum += scratch[groupIndex * 2u + 1u];
    }

    float paramNorm = sqrt(paramSum);
    float gradNorm = sqrt(gradSum);
    float trust = 1.0f;

    if (paramNorm > 0.0f && gradNorm > 0.0f) {
        trust = larsTrustCoeff * paramNorm /
            (gradNorm + larsWeightDecay * paramNorm + larsEpsilon);
    }

    float param = Storage::load(params, index);
    float decayed = Storage::load(gradients, index) + larsWeightDecay * param;
    momentum[index] = larsMomentum * momentum[index] + decayed;
    Storage::store(out, index, param - larsLearningRate * trust * momentum[index]);
}

#define OPTIMIZER4_KERNEL(name, storage, scalar, body) \
kernel void name( \
    device const scalar* params [[buffer(0)]], \
    device const scalar* gradients [[buffer(1)]], \
    device float* first [[buffer(2)]], \
    device float* second [[buffer(3)]], \
    device scalar* out [[buffer(4)]], \
    constant uint& count [[buffer(5)]], \
    uint index [[thread_position_in_grid]] \
) { \
    body<storage, scalar>(params, gradients, first, second, out, count, index); \
}

#define OPTIMIZER3_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* params [[buffer(0)]], \
    device const scalar* gradients [[buffer(1)]], \
    device float* state [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& count [[buffer(4)]], \
    constant uint& operation [[buffer(5)]], \
    uint index [[thread_position_in_grid]] \
) { \
    optimizer3_step_kernel<storage, scalar>(params, gradients, state, out, count, operation, index); \
}

#define OPTIMIZER2_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* params [[buffer(0)]], \
    device const scalar* gradients [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint index [[thread_position_in_grid]] \
) { \
    lbfgs_step_kernel<storage, scalar>(params, gradients, out, count, index); \
}

#define HEBBIAN_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* weights [[buffer(0)]], \
    device const scalar* post [[buffer(1)]], \
    device const scalar* pre [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& postCount [[buffer(4)]], \
    constant uint& preCount [[buffer(5)]], \
    uint index [[thread_position_in_grid]] \
) { \
    hebbian_step_kernel<storage, scalar>(weights, post, pre, out, postCount, preCount, index); \
}

#define LARS_NORMS_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* params [[buffer(0)]], \
    device const scalar* gradients [[buffer(1)]], \
    device float* scratch [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint localIndex [[thread_position_in_threadgroup]], \
    uint groupIndex [[threadgroup_position_in_grid]] \
) { \
    threadgroup float paramSums[256]; \
    threadgroup float gradSums[256]; \
    lars_norms_kernel<storage, scalar>( \
        params, gradients, scratch, paramSums, gradSums, count, localIndex, groupIndex \
    ); \
}

#define LARS_STEP_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* params [[buffer(0)]], \
    device const scalar* gradients [[buffer(1)]], \
    device float* momentum [[buffer(2)]], \
    device const float* scratch [[buffer(3)]], \
    device scalar* out [[buffer(4)]], \
    constant uint& count [[buffer(5)]], \
    constant uint& groupCount [[buffer(6)]], \
    uint index [[thread_position_in_grid]] \
) { \
    lars_step_kernel<storage, scalar>(params, gradients, momentum, scratch, out, count, groupCount, index); \
}

OPTIMIZER4_KERNEL(adam_step_float32, Float32OptimizerStorage, float, adam_step_kernel)
OPTIMIZER4_KERNEL(adam_step_float16, Float16OptimizerStorage, half, adam_step_kernel)
OPTIMIZER4_KERNEL(adam_step_bfloat16, BFloat16OptimizerStorage, ushort, adam_step_kernel)
OPTIMIZER4_KERNEL(adamw_step_float32, Float32OptimizerStorage, float, adamw_step_kernel)
OPTIMIZER4_KERNEL(adamw_step_float16, Float16OptimizerStorage, half, adamw_step_kernel)
OPTIMIZER4_KERNEL(adamw_step_bfloat16, BFloat16OptimizerStorage, ushort, adamw_step_kernel)
OPTIMIZER4_KERNEL(adamax_step_float32, Float32OptimizerStorage, float, adamax_step_kernel)
OPTIMIZER4_KERNEL(adamax_step_float16, Float16OptimizerStorage, half, adamax_step_kernel)
OPTIMIZER4_KERNEL(adamax_step_bfloat16, BFloat16OptimizerStorage, ushort, adamax_step_kernel)

OPTIMIZER3_KERNEL(optimizer3_float32, Float32OptimizerStorage, float)
OPTIMIZER3_KERNEL(optimizer3_float16, Float16OptimizerStorage, half)
OPTIMIZER3_KERNEL(optimizer3_bfloat16, BFloat16OptimizerStorage, ushort)
OPTIMIZER2_KERNEL(lbfgs_step_float32, Float32OptimizerStorage, float)
OPTIMIZER2_KERNEL(lbfgs_step_float16, Float16OptimizerStorage, half)
OPTIMIZER2_KERNEL(lbfgs_step_bfloat16, BFloat16OptimizerStorage, ushort)
HEBBIAN_KERNEL(hebbian_step_float32, Float32OptimizerStorage, float)
HEBBIAN_KERNEL(hebbian_step_float16, Float16OptimizerStorage, half)
HEBBIAN_KERNEL(hebbian_step_bfloat16, BFloat16OptimizerStorage, ushort)
LARS_NORMS_KERNEL(lars_norms_float32, Float32OptimizerStorage, float)
LARS_NORMS_KERNEL(lars_norms_float16, Float16OptimizerStorage, half)
LARS_NORMS_KERNEL(lars_norms_bfloat16, BFloat16OptimizerStorage, ushort)
LARS_STEP_KERNEL(lars_step_float32, Float32OptimizerStorage, float)
LARS_STEP_KERNEL(lars_step_float16, Float16OptimizerStorage, half)
LARS_STEP_KERNEL(lars_step_bfloat16, BFloat16OptimizerStorage, ushort)
