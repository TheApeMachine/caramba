#include <metal_stdlib>

using namespace metal;

static inline void copy_bytes_block(
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

kernel void copy_bytes(
    device const uint4* inputVector [[buffer(0)]],
    device uint4* outVector [[buffer(1)]],
    constant uint& byteCount [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 16;

    if (base + 15 < byteCount) {
        outVector[index] = inputVector[index];
        return;
    }

    device const uchar* input = reinterpret_cast<device const uchar*>(inputVector);
    device uchar* out = reinterpret_cast<device uchar*>(outVector);
    copy_bytes_block(input, out, byteCount, base);
}

kernel void concat_bytes(
    device const uchar* left [[buffer(0)]],
    device const uchar* right [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant uint& leftBytes [[buffer(3)]],
    constant uint& totalBytes [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 16;

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

kernel void split2_bytes(
    device const uchar* input [[buffer(0)]],
    device uchar* left [[buffer(1)]],
    device uchar* right [[buffer(2)]],
    constant uint& leftBytes [[buffer(3)]],
    constant uint& totalBytes [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 16;

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

kernel void last_token_bytes(
    device const uchar* input [[buffer(0)]],
    device uchar* out [[buffer(1)]],
    constant uint& seq [[buffer(2)]],
    constant uint& hiddenBytes [[buffer(3)]],
    constant uint& outBytes [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 16;

    for (uint offset = 0; offset < 16; offset++) {
        uint outIndex = base + offset;

        if (outIndex >= outBytes) {
            continue;
        }

        uint batchIndex = outIndex / hiddenBytes;
        uint hiddenOffset = outIndex - batchIndex * hiddenBytes;
        uint inputIndex = (batchIndex * seq + (seq - 1)) * hiddenBytes + hiddenOffset;
        out[outIndex] = input[inputIndex];
    }
}

kernel void transpose2d_bytes(
    device const uchar* input [[buffer(0)]],
    device uchar* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& elementBytes [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    uint elementCount = rows * cols;

    if (index >= elementCount) {
        return;
    }

    uint row = index / cols;
    uint col = index - row * cols;
    uint inputByte = index * elementBytes;
    uint outByte = (col * rows + row) * elementBytes;

    for (uint byteOffset = 0; byteOffset < elementBytes; byteOffset++) {
        out[outByte + byteOffset] = input[inputByte + byteOffset];
    }
}

kernel void upsample_nearest2d_bytes(
    device const uchar* input [[buffer(0)]],
    device uchar* out [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& inHeight [[buffer(3)]],
    constant uint& inWidth [[buffer(4)]],
    constant uint& outHeight [[buffer(5)]],
    constant uint& outWidth [[buffer(6)]],
    constant uint& elementBytes [[buffer(7)]],
    constant uint& outElements [[buffer(8)]],
    uint index [[thread_position_in_grid]]
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
    uint inputByte = inputElement * elementBytes;
    uint outByte = index * elementBytes;

    for (uint byteOffset = 0; byteOffset < elementBytes; byteOffset++) {
        out[outByte + byteOffset] = input[inputByte + byteOffset];
    }
}
