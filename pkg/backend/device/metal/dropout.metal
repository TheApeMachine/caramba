#include <metal_stdlib>

using namespace metal;

#include "dropout_jump.h"

struct Float32DropoutStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16DropoutStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

static inline float dropout_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort dropout_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct BFloat16DropoutStorage {
    static float load(device const ushort* values, uint index) {
        return dropout_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = dropout_float_to_bf16(value);
    }
};

static inline uint dropout_seed_for_index(uint index, uint count, uint4 seed) {
    uint blockCount = count & ~3u;

    if (index < blockCount) {
        uint lane = index & 3u;
        uint step = (index >> 2u) + 1u;

        if (lane == 0u) {
            return dropout_xorshift_advance(seed.x, step);
        }

        if (lane == 1u) {
            return dropout_xorshift_advance(seed.y, step);
        }

        if (lane == 2u) {
            return dropout_xorshift_advance(seed.z, step);
        }

        return dropout_xorshift_advance(seed.w, step);
    }

    uint tailStep = (blockCount >> 2u) + (index - blockCount) + 1u;
    return dropout_xorshift_advance(seed.x, tailStep);
}

template <typename Storage, typename Scalar>
static inline void dropout_kernel(
    device const Scalar* input,
    device Scalar* out,
    constant uint& count,
    constant float& scale,
    constant uint& threshold,
    constant uint4& seed,
    uint index
) {
    if (index >= count) {
        return;
    }

    uint randomValue = dropout_seed_for_index(index, count, seed);
    float outValue = 0.0f;

    if (randomValue < threshold) {
        outValue = Storage::load(input, index) * scale;
    }

    Storage::store(out, index, outValue);
}

#define DROPOUT_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device scalar* out [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    constant float& scale [[buffer(3)]], \
    constant uint& threshold [[buffer(4)]], \
    constant uint4& seed [[buffer(5)]], \
    uint index [[thread_position_in_grid]] \
) { \
    dropout_kernel<storage, scalar>(input, out, count, scale, threshold, seed, index); \
}

DROPOUT_KERNEL(dropout_float32, Float32DropoutStorage, float)
DROPOUT_KERNEL(dropout_float16, Float16DropoutStorage, half)
DROPOUT_KERNEL(dropout_bfloat16, BFloat16DropoutStorage, ushort)
