#include <metal_stdlib>

using namespace metal;

static inline void copy_tail_bytes(
    device const uchar* input,
    device uchar* out,
    uint byteCount,
    uint base
) {
    for (uint offset = 0; offset < 16; offset++) {
        uint byteIndex = base + offset;

        if (byteIndex < byteCount) {
            out[byteIndex] = input[byteIndex];
        }
    }
}

static inline void copy_bytes_kernel(
    device const uint4* inputVector,
    device uint4* outVector,
    constant uint& byteCount,
    uint index
) {
    uint base = index * 16;

    if (base + 15 < byteCount) {
        outVector[index] = inputVector[index];
        return;
    }

    device const uchar* input = reinterpret_cast<device const uchar*>(inputVector);
    device uchar* out = reinterpret_cast<device uchar*>(outVector);
    copy_tail_bytes(input, out, byteCount, base);
}

static inline void concat_bytes_kernel(
    device const uint4* leftVector,
    device const uint4* rightVector,
    device uint4* outVector,
    constant uint& leftBytes,
    constant uint& totalBytes,
    uint index
) {
    uint base = index * 16;

    if (base + 15 < leftBytes) {
        outVector[index] = leftVector[index];
        return;
    }

    if (base >= leftBytes && base + 15 < totalBytes && leftBytes % 16 == 0) {
        outVector[index] = rightVector[(base - leftBytes) / 16];
        return;
    }

    device const uchar* left = reinterpret_cast<device const uchar*>(leftVector);
    device const uchar* right = reinterpret_cast<device const uchar*>(rightVector);
    device uchar* out = reinterpret_cast<device uchar*>(outVector);

    for (uint offset = 0; offset < 16; offset++) {
        uint outIndex = base + offset;

        if (outIndex >= totalBytes) {
            continue;
        }

        if (outIndex < leftBytes) {
            out[outIndex] = left[outIndex];
            continue;
        }

        out[outIndex] = right[outIndex - leftBytes];
    }
}

static inline void split2_bytes_kernel(
    device const uint4* inputVector,
    device uint4* leftVector,
    device uint4* rightVector,
    constant uint& leftBytes,
    constant uint& totalBytes,
    uint index
) {
    uint base = index * 16;

    if (base + 15 < leftBytes) {
        leftVector[index] = inputVector[index];
        return;
    }

    if (base >= leftBytes && base + 15 < totalBytes && leftBytes % 16 == 0) {
        rightVector[(base - leftBytes) / 16] = inputVector[index];
        return;
    }

    device const uchar* input = reinterpret_cast<device const uchar*>(inputVector);
    device uchar* left = reinterpret_cast<device uchar*>(leftVector);
    device uchar* right = reinterpret_cast<device uchar*>(rightVector);

    for (uint offset = 0; offset < 16; offset++) {
        uint inputIndex = base + offset;

        if (inputIndex >= totalBytes) {
            continue;
        }

        if (inputIndex < leftBytes) {
            left[inputIndex] = input[inputIndex];
            continue;
        }

        right[inputIndex - leftBytes] = input[inputIndex];
    }
}

static inline void last_token_bytes_kernel(
    device const uint4* inputVector,
    device uint4* outVector,
    constant uint& seq,
    constant uint& hiddenBytes,
    constant uint& outBytes,
    uint index
) {
    uint base = index * 16;
    uint batchIndex = base / hiddenBytes;
    uint hiddenOffset = base - batchIndex * hiddenBytes;
    uint inputBase = (batchIndex * seq + (seq - 1)) * hiddenBytes + hiddenOffset;

    if (base + 15 < outBytes && hiddenOffset + 15 < hiddenBytes && inputBase % 16 == 0) {
        outVector[index] = inputVector[inputBase / 16];
        return;
    }

    device const uchar* input = reinterpret_cast<device const uchar*>(inputVector);
    device uchar* out = reinterpret_cast<device uchar*>(outVector);

    for (uint offset = 0; offset < 16; offset++) {
        uint outIndex = base + offset;

        if (outIndex >= outBytes) {
            continue;
        }

        batchIndex = outIndex / hiddenBytes;
        hiddenOffset = outIndex - batchIndex * hiddenBytes;
        uint inputIndex = (batchIndex * seq + (seq - 1)) * hiddenBytes + hiddenOffset;
        out[outIndex] = input[inputIndex];
    }
}

template <typename Storage>
static inline void transpose2d_kernel(
    device const Storage* input,
    device Storage* out,
    constant uint& rows,
    constant uint& cols,
    uint index
) {
    uint elementCount = rows * cols;

    if (index >= elementCount) {
        return;
    }

    uint row = index / cols;
    uint col = index - row * cols;
    out[col * rows + row] = input[index];
}

template <typename Storage>
static inline void upsample_nearest2d_kernel(
    device const Storage* input,
    device Storage* out,
    constant uint& channels,
    constant uint& inHeight,
    constant uint& inWidth,
    constant uint& outHeight,
    constant uint& outWidth,
    constant uint& outElements,
    uint index
) {
    if (index >= outElements) {
        return;
    }

    uint outCol = index % outWidth;
    uint outRow = (index / outWidth) % outHeight;
    uint channel = (index / (outWidth * outHeight)) % channels;
    uint batch = index / (outWidth * outHeight * channels);
    uint inRow = outRow * inHeight / outHeight;
    uint inCol = outCol * inWidth / outWidth;
    uint inputElement = ((batch * channels + channel) * inHeight + inRow) * inWidth + inCol;
    out[index] = input[inputElement];
}

#define COPY_KERNEL(name) \
kernel void name( \
    device const uint4* inputVector [[buffer(0)]], \
    device uint4* outVector [[buffer(1)]], \
    constant uint& byteCount [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    copy_bytes_kernel(inputVector, outVector, byteCount, index); \
}

#define CONCAT_KERNEL(name) \
kernel void name( \
    device const uint4* leftVector [[buffer(0)]], \
    device const uint4* rightVector [[buffer(1)]], \
    device uint4* outVector [[buffer(2)]], \
    constant uint& leftBytes [[buffer(3)]], \
    constant uint& totalBytes [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    concat_bytes_kernel(leftVector, rightVector, outVector, leftBytes, totalBytes, index); \
}

#define SPLIT2_KERNEL(name) \
kernel void name( \
    device const uint4* inputVector [[buffer(0)]], \
    device uint4* leftVector [[buffer(1)]], \
    device uint4* rightVector [[buffer(2)]], \
    constant uint& leftBytes [[buffer(3)]], \
    constant uint& totalBytes [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    split2_bytes_kernel(inputVector, leftVector, rightVector, leftBytes, totalBytes, index); \
}

#define LAST_TOKEN_KERNEL(name) \
kernel void name( \
    device const uint4* inputVector [[buffer(0)]], \
    device uint4* outVector [[buffer(1)]], \
    constant uint& seq [[buffer(2)]], \
    constant uint& hiddenBytes [[buffer(3)]], \
    constant uint& outBytes [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    last_token_bytes_kernel(inputVector, outVector, seq, hiddenBytes, outBytes, index); \
}

#define TRANSPOSE2D_KERNEL(name, storage) \
kernel void name( \
    device const storage* input [[buffer(0)]], \
    device storage* out [[buffer(1)]], \
    constant uint& rows [[buffer(2)]], \
    constant uint& cols [[buffer(3)]], \
    uint index [[thread_position_in_grid]] \
) { \
    transpose2d_kernel<storage>(input, out, rows, cols, index); \
}

#define UPSAMPLE_NEAREST2D_KERNEL(name, storage) \
kernel void name( \
    device const storage* input [[buffer(0)]], \
    device storage* out [[buffer(1)]], \
    constant uint& channels [[buffer(2)]], \
    constant uint& inHeight [[buffer(3)]], \
    constant uint& inWidth [[buffer(4)]], \
    constant uint& outHeight [[buffer(5)]], \
    constant uint& outWidth [[buffer(6)]], \
    constant uint& outElements [[buffer(7)]], \
    uint index [[thread_position_in_grid]] \
) { \
    upsample_nearest2d_kernel<storage>( \
        input, out, channels, inHeight, inWidth, outHeight, outWidth, outElements, index \
    ); \
}

COPY_KERNEL(copy_float32)
COPY_KERNEL(copy_float16)
COPY_KERNEL(copy_bfloat16)

CONCAT_KERNEL(concat_float32)
CONCAT_KERNEL(concat_float16)
CONCAT_KERNEL(concat_bfloat16)

SPLIT2_KERNEL(split2_float32)
SPLIT2_KERNEL(split2_float16)
SPLIT2_KERNEL(split2_bfloat16)

LAST_TOKEN_KERNEL(last_token_float32)
LAST_TOKEN_KERNEL(last_token_float16)
LAST_TOKEN_KERNEL(last_token_bfloat16)

TRANSPOSE2D_KERNEL(transpose2d_float32, uint)
TRANSPOSE2D_KERNEL(transpose2d_float16, ushort)
TRANSPOSE2D_KERNEL(transpose2d_bfloat16, ushort)

UPSAMPLE_NEAREST2D_KERNEL(upsample_nearest2d_float32, uint)
UPSAMPLE_NEAREST2D_KERNEL(upsample_nearest2d_float16, ushort)
UPSAMPLE_NEAREST2D_KERNEL(upsample_nearest2d_bfloat16, ushort)
