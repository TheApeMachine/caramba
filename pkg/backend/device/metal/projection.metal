#include <metal_stdlib>

using namespace metal;

constant uint projectionTileSize = 16;

static inline float projection_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort projection_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32ProjectionStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16ProjectionStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16ProjectionStorage {
    static float load(device const ushort* values, uint index) {
        return projection_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = projection_float_to_bf16(value);
    }
};

template <typename Storage, typename Scalar>
static inline void linear_tiled(
    device const Scalar* input,
    device const Scalar* weight,
    device const Scalar* bias,
    device Scalar* out,
    threadgroup float* inputTile,
    threadgroup float* weightTile,
    constant uint& batch,
    constant uint& inner,
    constant uint& outDim,
    uint2 localPosition,
    uint2 groupPosition
) {
    uint batchIndex = groupPosition.y * projectionTileSize + localPosition.y;
    uint outIndex = groupPosition.x * projectionTileSize + localPosition.x;
    uint localOffset = localPosition.y * projectionTileSize + localPosition.x;
    float accumulator = outIndex < outDim ? Storage::load(bias, outIndex) : 0.0f;

    for (uint tileStart = 0; tileStart < inner; tileStart += projectionTileSize) {
        uint inputInner = tileStart + localPosition.x;
        uint weightInner = tileStart + localPosition.y;

        inputTile[localOffset] = batchIndex < batch && inputInner < inner ?
            Storage::load(input, batchIndex * inner + inputInner) : 0.0f;
        weightTile[localOffset] = outIndex < outDim && weightInner < inner ?
            Storage::load(weight, outIndex * inner + weightInner) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tileIndex = 0; tileIndex < projectionTileSize; tileIndex++) {
            accumulator += inputTile[localPosition.y * projectionTileSize + tileIndex] *
                weightTile[tileIndex * projectionTileSize + localPosition.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (batchIndex < batch && outIndex < outDim) {
        Storage::store(out, batchIndex * outDim + outIndex, accumulator);
    }
}

template <typename Storage, typename Scalar>
static inline void fused_qkv_tiled(
    device const Scalar* input,
    device const Scalar* weight,
    device const Scalar* bias,
    device Scalar* query,
    device Scalar* key,
    device Scalar* value,
    threadgroup float* inputTile,
    threadgroup float* queryWeightTile,
    threadgroup float* keyWeightTile,
    threadgroup float* valueWeightTile,
    constant uint& batch,
    constant uint& inner,
    constant uint& outDim,
    uint2 localPosition,
    uint2 groupPosition
) {
    uint batchIndex = groupPosition.y * projectionTileSize + localPosition.y;
    uint outIndex = groupPosition.x * projectionTileSize + localPosition.x;
    uint localOffset = localPosition.y * projectionTileSize + localPosition.x;
    float queryAccumulator = outIndex < outDim ? Storage::load(bias, outIndex) : 0.0f;
    float keyAccumulator = outIndex < outDim ? Storage::load(bias, outDim + outIndex) : 0.0f;
    float valueAccumulator = outIndex < outDim ? Storage::load(bias, 2 * outDim + outIndex) : 0.0f;

    for (uint tileStart = 0; tileStart < inner; tileStart += projectionTileSize) {
        uint inputInner = tileStart + localPosition.x;
        uint weightInner = tileStart + localPosition.y;

        inputTile[localOffset] = batchIndex < batch && inputInner < inner ?
            Storage::load(input, batchIndex * inner + inputInner) : 0.0f;
        queryWeightTile[localOffset] = outIndex < outDim && weightInner < inner ?
            Storage::load(weight, outIndex * inner + weightInner) : 0.0f;
        keyWeightTile[localOffset] = outIndex < outDim && weightInner < inner ?
            Storage::load(weight, (outDim + outIndex) * inner + weightInner) : 0.0f;
        valueWeightTile[localOffset] = outIndex < outDim && weightInner < inner ?
            Storage::load(weight, (2 * outDim + outIndex) * inner + weightInner) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tileIndex = 0; tileIndex < projectionTileSize; tileIndex++) {
            float inputValue = inputTile[localPosition.y * projectionTileSize + tileIndex];
            uint weightOffset = tileIndex * projectionTileSize + localPosition.x;
            queryAccumulator += inputValue * queryWeightTile[weightOffset];
            keyAccumulator += inputValue * keyWeightTile[weightOffset];
            valueAccumulator += inputValue * valueWeightTile[weightOffset];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (batchIndex < batch && outIndex < outDim) {
        uint outputIndex = batchIndex * outDim + outIndex;
        Storage::store(query, outputIndex, queryAccumulator);
        Storage::store(key, outputIndex, keyAccumulator);
        Storage::store(value, outputIndex, valueAccumulator);
    }
}

template <typename Storage, typename Scalar>
static inline void lora_merge_kernel(
    device const Scalar* baseWeight,
    device const Scalar* loraA,
    device const Scalar* loraB,
    device Scalar* out,
    constant uint& outDim,
    constant uint& rank,
    constant uint& inner,
    uint2 groupPosition,
    uint2 localPosition
) {
    uint innerIndex = groupPosition.x * projectionTileSize + localPosition.x;
    uint outIndex = groupPosition.y * projectionTileSize + localPosition.y;

    if (outIndex >= outDim || innerIndex >= inner) {
        return;
    }

    uint outputIndex = outIndex * inner + innerIndex;
    float accumulator = Storage::load(baseWeight, outputIndex);

    for (uint rankIndex = 0; rankIndex < rank; rankIndex++) {
        accumulator += Storage::load(loraA, outIndex * rank + rankIndex) *
            Storage::load(loraB, rankIndex * inner + innerIndex);
    }

    Storage::store(out, outputIndex, accumulator);
}

template <typename Storage, typename Scalar>
static inline void lora_apply_stage1_kernel(
    device const Scalar* input,
    device const Scalar* loraB,
    device float* scratch,
    constant uint& batch,
    constant uint& inner,
    constant uint& rank,
    uint2 groupPosition,
    uint2 localPosition
) {
    uint rankIndex = groupPosition.x * projectionTileSize + localPosition.x;
    uint batchIndex = groupPosition.y * projectionTileSize + localPosition.y;

    if (batchIndex >= batch || rankIndex >= rank) {
        return;
    }

    float accumulator = 0.0f;

    for (uint innerIndex = 0; innerIndex < inner; innerIndex++) {
        accumulator += Storage::load(input, batchIndex * inner + innerIndex) *
            Storage::load(loraB, rankIndex * inner + innerIndex);
    }

    scratch[batchIndex * rank + rankIndex] = accumulator;
}

template <typename Storage, typename Scalar>
static inline void lora_apply_stage2_kernel(
    device const Scalar* baseOut,
    device const Scalar* loraA,
    device const float* scratch,
    device Scalar* out,
    constant uint& batch,
    constant uint& rank,
    constant uint& outDim,
    uint2 groupPosition,
    uint2 localPosition
) {
    uint outIndex = groupPosition.x * projectionTileSize + localPosition.x;
    uint batchIndex = groupPosition.y * projectionTileSize + localPosition.y;

    if (batchIndex >= batch || outIndex >= outDim) {
        return;
    }

    uint outputIndex = batchIndex * outDim + outIndex;
    float accumulator = Storage::load(baseOut, outputIndex);

    for (uint rankIndex = 0; rankIndex < rank; rankIndex++) {
        accumulator += Storage::load(loraA, outIndex * rank + rankIndex) *
            scratch[batchIndex * rank + rankIndex];
    }

    Storage::store(out, outputIndex, accumulator);
}

#define LINEAR_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* weight [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& batch [[buffer(4)]], \
    constant uint& inner [[buffer(5)]], \
    constant uint& outDim [[buffer(6)]], \
    uint2 localPosition [[thread_position_in_threadgroup]], \
    uint2 groupPosition [[threadgroup_position_in_grid]] \
) { \
    threadgroup float inputTile[256]; \
    threadgroup float weightTile[256]; \
    linear_tiled<storage, scalar>( \
        input, weight, bias, out, inputTile, weightTile, \
        batch, inner, outDim, localPosition, groupPosition \
    ); \
}

#define FUSED_QKV_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* weight [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* query [[buffer(3)]], \
    device scalar* key [[buffer(4)]], \
    device scalar* value [[buffer(5)]], \
    constant uint& batch [[buffer(6)]], \
    constant uint& inner [[buffer(7)]], \
    constant uint& outDim [[buffer(8)]], \
    uint2 localPosition [[thread_position_in_threadgroup]], \
    uint2 groupPosition [[threadgroup_position_in_grid]] \
) { \
    threadgroup float inputTile[256]; \
    threadgroup float queryWeightTile[256]; \
    threadgroup float keyWeightTile[256]; \
    threadgroup float valueWeightTile[256]; \
    fused_qkv_tiled<storage, scalar>( \
        input, weight, bias, query, key, value, \
        inputTile, queryWeightTile, keyWeightTile, valueWeightTile, \
        batch, inner, outDim, localPosition, groupPosition \
    ); \
}

#define LORA_MERGE_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* baseWeight [[buffer(0)]], \
    device const scalar* loraA [[buffer(1)]], \
    device const scalar* loraB [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& outDim [[buffer(4)]], \
    constant uint& rank [[buffer(5)]], \
    constant uint& inner [[buffer(6)]], \
    uint2 localPosition [[thread_position_in_threadgroup]], \
    uint2 groupPosition [[threadgroup_position_in_grid]] \
) { \
    lora_merge_kernel<storage, scalar>( \
        baseWeight, loraA, loraB, out, outDim, rank, inner, groupPosition, localPosition \
    ); \
}

#define LORA_APPLY_STAGE1_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* loraB [[buffer(1)]], \
    device float* scratch [[buffer(2)]], \
    constant uint& batch [[buffer(3)]], \
    constant uint& inner [[buffer(4)]], \
    constant uint& rank [[buffer(5)]], \
    uint2 localPosition [[thread_position_in_threadgroup]], \
    uint2 groupPosition [[threadgroup_position_in_grid]] \
) { \
    lora_apply_stage1_kernel<storage, scalar>( \
        input, loraB, scratch, batch, inner, rank, groupPosition, localPosition \
    ); \
}

#define LORA_APPLY_STAGE2_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* baseOut [[buffer(0)]], \
    device const scalar* loraA [[buffer(1)]], \
    device const float* scratch [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& batch [[buffer(4)]], \
    constant uint& rank [[buffer(5)]], \
    constant uint& outDim [[buffer(6)]], \
    uint2 localPosition [[thread_position_in_threadgroup]], \
    uint2 groupPosition [[threadgroup_position_in_grid]] \
) { \
    lora_apply_stage2_kernel<storage, scalar>( \
        baseOut, loraA, scratch, out, batch, rank, outDim, groupPosition, localPosition \
    ); \
}

LINEAR_KERNEL(linear_float32, Float32ProjectionStorage, float)
LINEAR_KERNEL(linear_float16, Float16ProjectionStorage, half)
LINEAR_KERNEL(linear_bfloat16, BFloat16ProjectionStorage, ushort)

FUSED_QKV_KERNEL(fused_qkv_float32, Float32ProjectionStorage, float)
FUSED_QKV_KERNEL(fused_qkv_float16, Float16ProjectionStorage, half)
FUSED_QKV_KERNEL(fused_qkv_bfloat16, BFloat16ProjectionStorage, ushort)

LORA_MERGE_KERNEL(lora_merge_float32, Float32ProjectionStorage, float)
LORA_MERGE_KERNEL(lora_merge_float16, Float16ProjectionStorage, half)
LORA_MERGE_KERNEL(lora_merge_bfloat16, BFloat16ProjectionStorage, ushort)

LORA_APPLY_STAGE1_KERNEL(lora_apply_stage1_float32, Float32ProjectionStorage, float)
LORA_APPLY_STAGE1_KERNEL(lora_apply_stage1_float16, Float16ProjectionStorage, half)
LORA_APPLY_STAGE1_KERNEL(lora_apply_stage1_bfloat16, BFloat16ProjectionStorage, ushort)

LORA_APPLY_STAGE2_KERNEL(lora_apply_stage2_float32, Float32ProjectionStorage, float)
LORA_APPLY_STAGE2_KERNEL(lora_apply_stage2_float16, Float16ProjectionStorage, half)
LORA_APPLY_STAGE2_KERNEL(lora_apply_stage2_bfloat16, BFloat16ProjectionStorage, ushort)
