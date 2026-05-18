#include <metal_stdlib>

using namespace metal;

static inline float transformer_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort transformer_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32TransformerStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16TransformerStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16TransformerStorage {
    static float load(device const ushort* values, uint index) {
        return transformer_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = transformer_float_to_bf16(value);
    }
};

static inline void set_transformer_error(device atomic_uint* errorFlag) {
    atomic_store_explicit(errorFlag, 1u, memory_order_relaxed);
}

template <typename Storage, typename Scalar>
static inline void embedding_lookup_kernel(
    device const Scalar* table,
    device const int* indices,
    device Scalar* out,
    device atomic_uint* errorFlag,
    constant uint& vocab,
    constant uint& hidden,
    constant uint& indexCount,
    uint outputIndex
) {
    uint total = indexCount * hidden;

    if (outputIndex >= total) {
        return;
    }

    uint tokenOffset = outputIndex / hidden;
    uint hiddenOffset = outputIndex - tokenOffset * hidden;
    int tokenID = indices[tokenOffset];

    if (tokenID < 0 || uint(tokenID) >= vocab) {
        set_transformer_error(errorFlag);
        return;
    }

    out[outputIndex] = table[uint(tokenID) * hidden + hiddenOffset];
}

template <typename Storage, typename Scalar>
static inline void embedding_bag_kernel(
    device const Scalar* table,
    device const int* indices,
    device const int* offsets,
    device Scalar* out,
    device atomic_uint* errorFlag,
    constant uint& vocab,
    constant uint& hidden,
    constant uint& indexCount,
    constant uint& bagCount,
    uint outputIndex
) {
    uint total = bagCount * hidden;

    if (outputIndex >= total) {
        return;
    }

    uint bagIndex = outputIndex / hidden;
    uint hiddenOffset = outputIndex - bagIndex * hidden;
    int start = offsets[bagIndex];
    int end = bagIndex + 1 < bagCount ? offsets[bagIndex + 1] : int(indexCount);

    if (start < 0 || end < start || uint(end) > indexCount) {
        set_transformer_error(errorFlag);
        return;
    }

    float accumulator = 0.0f;

    for (int indexCursor = start; indexCursor < end; indexCursor++) {
        int tokenID = indices[indexCursor];

        if (tokenID < 0 || uint(tokenID) >= vocab) {
            set_transformer_error(errorFlag);
            return;
        }

        accumulator += Storage::load(table, uint(tokenID) * hidden + hiddenOffset);
    }

    Storage::store(out, outputIndex, accumulator);
}

template <typename Storage, typename Scalar>
static inline void apply_mask_kernel(
    device const Scalar* input,
    device const Scalar* mask,
    device Scalar* out,
    constant uint& count,
    uint index
) {
    if (index >= count) {
        return;
    }

    Storage::store(out, index, Storage::load(input, index) + Storage::load(mask, index));
}

template <typename Storage, typename Scalar>
static inline void causal_mask_kernel(
    device Scalar* out,
    constant uint& rows,
    constant uint& cols,
    uint index
) {
    uint count = rows * cols;

    if (index >= count) {
        return;
    }

    uint row = index / cols;
    uint col = index - row * cols;
    float value = col > row ? -INFINITY : 0.0f;
    Storage::store(out, index, value);
}

template <typename Storage, typename Scalar>
static inline void alibi_bias_kernel(
    device const Scalar* scores,
    device const Scalar* slope,
    device Scalar* out,
    constant uint& rows,
    constant uint& cols,
    uint index
) {
    uint count = rows * cols;

    if (index >= count) {
        return;
    }

    uint row = index / cols;
    uint col = index - row * cols;
    float value = Storage::load(scores, index);

    if (row >= col) {
        value -= Storage::load(slope, 0) * float(row - col);
    }

    Storage::store(out, index, value);
}

#define EMBEDDING_LOOKUP_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* table [[buffer(0)]], \
    device const int* indices [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    device atomic_uint* errorFlag [[buffer(3)]], \
    constant uint& vocab [[buffer(4)]], \
    constant uint& hidden [[buffer(5)]], \
    constant uint& indexCount [[buffer(6)]], \
    uint index [[thread_position_in_grid]] \
) { \
    embedding_lookup_kernel<storage, scalar>( \
        table, indices, out, errorFlag, vocab, hidden, indexCount, index \
    ); \
}

#define EMBEDDING_BAG_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* table [[buffer(0)]], \
    device const int* indices [[buffer(1)]], \
    device const int* offsets [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    device atomic_uint* errorFlag [[buffer(4)]], \
    constant uint& vocab [[buffer(5)]], \
    constant uint& hidden [[buffer(6)]], \
    constant uint& indexCount [[buffer(7)]], \
    constant uint& bagCount [[buffer(8)]], \
    uint index [[thread_position_in_grid]] \
) { \
    embedding_bag_kernel<storage, scalar>( \
        table, indices, offsets, out, errorFlag, vocab, hidden, indexCount, bagCount, index \
    ); \
}

#define APPLY_MASK_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* mask [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint index [[thread_position_in_grid]] \
) { \
    apply_mask_kernel<storage, scalar>(input, mask, out, count, index); \
}

#define CAUSAL_MASK_KERNEL(name, storage, scalar) \
kernel void name( \
    device scalar* out [[buffer(0)]], \
    constant uint& rows [[buffer(1)]], \
    constant uint& cols [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    causal_mask_kernel<storage, scalar>(out, rows, cols, index); \
}

#define ALIBI_BIAS_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* scores [[buffer(0)]], \
    device const scalar* slope [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& rows [[buffer(3)]], \
    constant uint& cols [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    alibi_bias_kernel<storage, scalar>(scores, slope, out, rows, cols, index); \
}

EMBEDDING_LOOKUP_KERNEL(embedding_lookup_float32, Float32TransformerStorage, float)
EMBEDDING_LOOKUP_KERNEL(embedding_lookup_float16, Float16TransformerStorage, half)
EMBEDDING_LOOKUP_KERNEL(embedding_lookup_bfloat16, BFloat16TransformerStorage, ushort)

EMBEDDING_BAG_KERNEL(embedding_bag_float32, Float32TransformerStorage, float)
EMBEDDING_BAG_KERNEL(embedding_bag_float16, Float16TransformerStorage, half)
EMBEDDING_BAG_KERNEL(embedding_bag_bfloat16, BFloat16TransformerStorage, ushort)

APPLY_MASK_KERNEL(apply_mask_float32, Float32TransformerStorage, float)
APPLY_MASK_KERNEL(apply_mask_float16, Float16TransformerStorage, half)
APPLY_MASK_KERNEL(apply_mask_bfloat16, BFloat16TransformerStorage, ushort)

CAUSAL_MASK_KERNEL(causal_mask_float32, Float32TransformerStorage, float)
CAUSAL_MASK_KERNEL(causal_mask_float16, Float16TransformerStorage, half)
CAUSAL_MASK_KERNEL(causal_mask_bfloat16, BFloat16TransformerStorage, ushort)

ALIBI_BIAS_KERNEL(alibi_bias_float32, Float32TransformerStorage, float)
ALIBI_BIAS_KERNEL(alibi_bias_float16, Float16TransformerStorage, half)
ALIBI_BIAS_KERNEL(alibi_bias_bfloat16, BFloat16TransformerStorage, ushort)
