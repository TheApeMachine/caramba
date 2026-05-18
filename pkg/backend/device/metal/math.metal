#include <metal_stdlib>

using namespace metal;

constant uint mathThreadCount = 256;

static inline float math_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort math_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32MathStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16MathStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16MathStorage {
    static float load(device const ushort* values, uint index) {
        return math_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = math_float_to_bf16(value);
    }
};

template <typename Storage, typename Scalar>
static inline void inv_sqrt_dim_scale_kernel(
    device const Scalar* input,
    device const int* dim,
    device Scalar* out,
    device atomic_uint* errorFlag,
    constant uint& count,
    uint index
) {
    int scaleDim = dim[0];
    if (scaleDim <= 0) {
        if (index == 0) {
            atomic_store_explicit(errorFlag, 1u, memory_order_relaxed);
        }
        return;
    }

    if (index >= count) {
        return;
    }

    float scale = 1.0f / sqrt(float(scaleDim));
    Storage::store(out, index, Storage::load(input, index) * scale);
}

template <typename Storage, typename Scalar>
static inline void logsumexp_row(
    device const Scalar* input,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& cols,
    uint row,
    uint threadIndex
) {
    uint rowOffset = row * cols;
    float localMax = -3.4028234663852886e38f;

    for (uint col = threadIndex; col < cols; col += mathThreadCount) {
        localMax = max(localMax, Storage::load(input, rowOffset + col));
    }

    reduction[threadIndex] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = mathThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] = max(reduction[threadIndex], reduction[threadIndex + stride]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float maximum = reduction[0];
    float localSum = 0.0f;

    for (uint col = threadIndex; col < cols; col += mathThreadCount) {
        localSum += exp(Storage::load(input, rowOffset + col) - maximum);
    }

    reduction[threadIndex] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = mathThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        Storage::store(out, row, maximum + log(reduction[0]));
    }
}

template <typename Storage, typename Scalar>
static inline void outer_kernel(
    device const Scalar* left,
    device const Scalar* right,
    device Scalar* out,
    constant uint& cols,
    constant uint& count,
    uint index
) {
    if (index >= count) {
        return;
    }

    uint row = index / cols;
    uint col = index - row * cols;
    Storage::store(out, index, Storage::load(left, row) * Storage::load(right, col));
}

#define INV_SQRT_DIM_SCALE_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const int* dim [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    device atomic_uint* errorFlag [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    inv_sqrt_dim_scale_kernel<storage, scalar>(input, dim, out, errorFlag, count, index); \
}

#define LOGSUMEXP_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device scalar* out [[buffer(1)]], \
    constant uint& cols [[buffer(2)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    logsumexp_row<storage, scalar>(input, out, reduction, cols, row, threadIndex); \
}

#define OUTER_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* left [[buffer(0)]], \
    device const scalar* right [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& cols [[buffer(3)]], \
    constant uint& count [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    outer_kernel<storage, scalar>(left, right, out, cols, count, index); \
}

INV_SQRT_DIM_SCALE_KERNEL(inv_sqrt_dim_scale_float32, Float32MathStorage, float)
LOGSUMEXP_KERNEL(logsumexp_float32, Float32MathStorage, float)
OUTER_KERNEL(outer_float32, Float32MathStorage, float)

INV_SQRT_DIM_SCALE_KERNEL(inv_sqrt_dim_scale_float16, Float16MathStorage, half)
LOGSUMEXP_KERNEL(logsumexp_float16, Float16MathStorage, half)
OUTER_KERNEL(outer_float16, Float16MathStorage, half)

INV_SQRT_DIM_SCALE_KERNEL(inv_sqrt_dim_scale_bfloat16, BFloat16MathStorage, ushort)
LOGSUMEXP_KERNEL(logsumexp_bfloat16, BFloat16MathStorage, ushort)
OUTER_KERNEL(outer_bfloat16, BFloat16MathStorage, ushort)
