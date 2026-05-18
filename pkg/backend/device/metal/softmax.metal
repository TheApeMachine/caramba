#include <metal_stdlib>

using namespace metal;

constant uint softmaxThreadCount = 256;

static inline float bf16_to_float_softmax(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort float_to_bf16_softmax(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32SoftmaxStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16SoftmaxStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16SoftmaxStorage {
    static float load(device const ushort* values, uint index) {
        return bf16_to_float_softmax(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = float_to_bf16_softmax(value);
    }
};

template <typename Storage, typename Scalar>
static inline void softmax_rows(
    device const Scalar* input,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& cols,
    uint row,
    uint threadIndex
) {
    uint rowOffset = row * cols;
    float localMax = -3.4028234663852886e38f;

    for (uint col = threadIndex; col < cols; col += softmaxThreadCount) {
        localMax = max(localMax, Storage::load(input, rowOffset + col));
    }

    reduction[threadIndex] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = softmaxThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] = max(reduction[threadIndex], reduction[threadIndex + stride]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float maximum = reduction[0];
    float localSum = 0.0f;

    for (uint col = threadIndex; col < cols; col += softmaxThreadCount) {
        localSum += exp(Storage::load(input, rowOffset + col) - maximum);
    }

    reduction[threadIndex] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = softmaxThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum = reduction[0];

    for (uint col = threadIndex; col < cols; col += softmaxThreadCount) {
        float value = sum == 0.0f ? 0.0f : exp(Storage::load(input, rowOffset + col) - maximum) / sum;
        Storage::store(out, rowOffset + col, value);
    }
}

#define SOFTMAX_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device scalar* out [[buffer(1)]], \
    constant uint& cols [[buffer(2)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    softmax_rows<storage, scalar>(input, out, reduction, cols, row, threadIndex); \
}

SOFTMAX_KERNEL(softmax_float32, Float32SoftmaxStorage, float)
SOFTMAX_KERNEL(softmax_float16, Float16SoftmaxStorage, half)
SOFTMAX_KERNEL(softmax_bfloat16, BFloat16SoftmaxStorage, ushort)
