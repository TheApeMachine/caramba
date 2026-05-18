#include <metal_stdlib>

using namespace metal;

constant uint matmulTileSize = 16;

static inline float bf16_to_float_matmul(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort float_to_bf16_matmul(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32MatMulStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16MatMulStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16MatMulStorage {
    static float load(device const ushort* values, uint index) {
        return bf16_to_float_matmul(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = float_to_bf16_matmul(value);
    }
};

template <typename Storage, typename Scalar>
static inline void matmul_tiled(
    device const Scalar* left,
    device const Scalar* right,
    device Scalar* out,
    threadgroup float* leftTile,
    threadgroup float* rightTile,
    constant uint& rows,
    constant uint& inner,
    constant uint& cols,
    uint2 localPosition,
    uint2 groupPosition
) {
    uint row = groupPosition.y * matmulTileSize + localPosition.y;
    uint col = groupPosition.x * matmulTileSize + localPosition.x;
    uint localOffset = localPosition.y * matmulTileSize + localPosition.x;
    float accumulator = 0.0f;

    for (uint tileStart = 0; tileStart < inner; tileStart += matmulTileSize) {
        uint leftInner = tileStart + localPosition.x;
        uint rightInner = tileStart + localPosition.y;

        leftTile[localOffset] =
            row < rows && leftInner < inner ? Storage::load(left, row * inner + leftInner) : 0.0f;
        rightTile[localOffset] =
            rightInner < inner && col < cols ? Storage::load(right, rightInner * cols + col) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tileIndex = 0; tileIndex < matmulTileSize; tileIndex++) {
            accumulator += leftTile[localPosition.y * matmulTileSize + tileIndex] *
                rightTile[tileIndex * matmulTileSize + localPosition.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < rows && col < cols) {
        Storage::store(out, row * cols + col, accumulator);
    }
}

template <typename Storage, typename Scalar>
static inline void matmul_add_tiled(
    device const Scalar* left,
    device const Scalar* right,
    device const Scalar* bias,
    device Scalar* out,
    threadgroup float* leftTile,
    threadgroup float* rightTile,
    constant uint& rows,
    constant uint& inner,
    constant uint& cols,
    uint2 localPosition,
    uint2 groupPosition
) {
    uint row = groupPosition.y * matmulTileSize + localPosition.y;
    uint col = groupPosition.x * matmulTileSize + localPosition.x;
    uint localOffset = localPosition.y * matmulTileSize + localPosition.x;
    float accumulator = col < cols ? Storage::load(bias, col) : 0.0f;

    for (uint tileStart = 0; tileStart < inner; tileStart += matmulTileSize) {
        uint leftInner = tileStart + localPosition.x;
        uint rightInner = tileStart + localPosition.y;

        leftTile[localOffset] =
            row < rows && leftInner < inner ? Storage::load(left, row * inner + leftInner) : 0.0f;
        rightTile[localOffset] =
            rightInner < inner && col < cols ? Storage::load(right, rightInner * cols + col) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tileIndex = 0; tileIndex < matmulTileSize; tileIndex++) {
            accumulator += leftTile[localPosition.y * matmulTileSize + tileIndex] *
                rightTile[tileIndex * matmulTileSize + localPosition.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < rows && col < cols) {
        Storage::store(out, row * cols + col, accumulator);
    }
}

#define MATMUL_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* left [[buffer(0)]], \
    device const scalar* right [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& rows [[buffer(3)]], \
    constant uint& inner [[buffer(4)]], \
    constant uint& cols [[buffer(5)]], \
    uint2 localPosition [[thread_position_in_threadgroup]], \
    uint2 groupPosition [[threadgroup_position_in_grid]] \
) { \
    threadgroup float leftTile[256]; \
    threadgroup float rightTile[256]; \
    matmul_tiled<storage, scalar>( \
        left, right, out, leftTile, rightTile, rows, inner, cols, localPosition, groupPosition \
    ); \
}

#define MATMUL_ADD_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* left [[buffer(0)]], \
    device const scalar* right [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& rows [[buffer(4)]], \
    constant uint& inner [[buffer(5)]], \
    constant uint& cols [[buffer(6)]], \
    uint2 localPosition [[thread_position_in_threadgroup]], \
    uint2 groupPosition [[threadgroup_position_in_grid]] \
) { \
    threadgroup float leftTile[256]; \
    threadgroup float rightTile[256]; \
    matmul_add_tiled<storage, scalar>( \
        left, right, bias, out, leftTile, rightTile, \
        rows, inner, cols, localPosition, groupPosition \
    ); \
}

MATMUL_KERNEL(matmul_float32, Float32MatMulStorage, float)
MATMUL_KERNEL(matmul_float16, Float16MatMulStorage, half)
MATMUL_KERNEL(matmul_bfloat16, BFloat16MatMulStorage, ushort)

MATMUL_ADD_KERNEL(matmul_add_float32, Float32MatMulStorage, float)
MATMUL_ADD_KERNEL(matmul_add_float16, Float16MatMulStorage, half)
MATMUL_ADD_KERNEL(matmul_add_bfloat16, BFloat16MatMulStorage, ushort)
