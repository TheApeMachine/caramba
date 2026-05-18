#include <metal_stdlib>

using namespace metal;

static inline float vision_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort vision_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32VisionStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16VisionStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16VisionStorage {
    static float load(device const ushort* values, uint index) {
        return vision_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = vision_float_to_bf16(value);
    }
};

template <typename Storage, typename Scalar>
static inline void conv1d_kernel(
    device const Scalar* input,
    device const Scalar* weight,
    device const Scalar* bias,
    device Scalar* out,
    constant uint& batch,
    constant uint& inChannels,
    constant uint& inLength,
    constant uint& outChannels,
    constant uint& kernelLength,
    constant uint& outLength,
    uint index
) {
    uint count = batch * outChannels * outLength;

    if (index >= count) {
        return;
    }

    uint outPosition = index % outLength;
    uint outChannel = (index / outLength) % outChannels;
    uint batchIndex = index / (outLength * outChannels);
    float accumulator = Storage::load(bias, outChannel);

    for (uint inChannel = 0; inChannel < inChannels; inChannel++) {
        for (uint kernelPosition = 0; kernelPosition < kernelLength; kernelPosition++) {
            uint inputPosition = outPosition + kernelPosition;

            if (inputPosition >= inLength) {
                continue;
            }

            uint inputIndex = (batchIndex * inChannels + inChannel) * inLength + inputPosition;
            uint weightIndex = (outChannel * inChannels + inChannel) *
                kernelLength + kernelPosition;
            accumulator += Storage::load(input, inputIndex) * Storage::load(weight, weightIndex);
        }
    }

    Storage::store(out, index, accumulator);
}

template <typename Storage, typename Scalar>
static inline void conv2d_kernel(
    device const Scalar* input,
    device const Scalar* weight,
    device const Scalar* bias,
    device Scalar* out,
    constant uint& batch,
    constant uint& inChannels,
    constant uint& inHeight,
    constant uint& inWidth,
    constant uint& outChannels,
    constant uint& kernelHeight,
    constant uint& kernelWidth,
    constant uint& outHeight,
    constant uint& outWidth,
    uint index
) {
    uint count = batch * outChannels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    uint outCol = index % outWidth;
    uint outRow = (index / outWidth) % outHeight;
    uint outChannel = (index / (outWidth * outHeight)) % outChannels;
    uint batchIndex = index / (outWidth * outHeight * outChannels);
    float accumulator = Storage::load(bias, outChannel);

    for (uint inChannel = 0; inChannel < inChannels; inChannel++) {
        for (uint kernelRow = 0; kernelRow < kernelHeight; kernelRow++) {
            uint inRow = outRow + kernelRow;

            if (inRow >= inHeight) {
                continue;
            }

            for (uint kernelCol = 0; kernelCol < kernelWidth; kernelCol++) {
                uint inCol = outCol + kernelCol;

                if (inCol >= inWidth) {
                    continue;
                }

                uint inputIndex = ((batchIndex * inChannels + inChannel) * inHeight + inRow) *
                    inWidth + inCol;
                uint weightIndex = ((outChannel * inChannels + inChannel) * kernelHeight +
                    kernelRow) * kernelWidth + kernelCol;
                accumulator += Storage::load(input, inputIndex) * Storage::load(weight, weightIndex);
            }
        }
    }

    Storage::store(out, index, accumulator);
}

template <typename Storage, typename Scalar>
static inline void conv3d_kernel(
    device const Scalar* input,
    device const Scalar* weight,
    device const Scalar* bias,
    device Scalar* out,
    constant uint& batch,
    constant uint& inChannels,
    constant uint& inDepth,
    constant uint& inHeight,
    constant uint& inWidth,
    constant uint& outChannels,
    constant uint& kernelDepth,
    constant uint& kernelHeight,
    constant uint& kernelWidth,
    constant uint& outDepth,
    constant uint& outHeight,
    constant uint& outWidth,
    uint index
) {
    uint count = batch * outChannels * outDepth * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    uint outCol = index % outWidth;
    uint outRow = (index / outWidth) % outHeight;
    uint outPlane = (index / (outWidth * outHeight)) % outDepth;
    uint outChannel = (index / (outWidth * outHeight * outDepth)) % outChannels;
    uint batchIndex = index / (outWidth * outHeight * outDepth * outChannels);
    float accumulator = Storage::load(bias, outChannel);

    for (uint inChannel = 0; inChannel < inChannels; inChannel++) {
        for (uint kernelPlane = 0; kernelPlane < kernelDepth; kernelPlane++) {
            uint inPlane = outPlane + kernelPlane;

            if (inPlane >= inDepth) {
                continue;
            }

            for (uint kernelRow = 0; kernelRow < kernelHeight; kernelRow++) {
                uint inRow = outRow + kernelRow;

                if (inRow >= inHeight) {
                    continue;
                }

                for (uint kernelCol = 0; kernelCol < kernelWidth; kernelCol++) {
                    uint inCol = outCol + kernelCol;

                    if (inCol >= inWidth) {
                        continue;
                    }

                    uint inputIndex = (((batchIndex * inChannels + inChannel) * inDepth +
                        inPlane) * inHeight + inRow) * inWidth + inCol;
                    uint weightIndex = (((outChannel * inChannels + inChannel) * kernelDepth +
                        kernelPlane) * kernelHeight + kernelRow) * kernelWidth + kernelCol;
                    accumulator += Storage::load(input, inputIndex) *
                        Storage::load(weight, weightIndex);
                }
            }
        }
    }

    Storage::store(out, index, accumulator);
}

template <typename Storage, typename Scalar>
static inline void conv_transpose2d_kernel(
    device const Scalar* input,
    device const Scalar* weight,
    device const Scalar* bias,
    device Scalar* out,
    constant uint& batch,
    constant uint& inChannels,
    constant uint& inHeight,
    constant uint& inWidth,
    constant uint& outChannels,
    constant uint& kernelHeight,
    constant uint& kernelWidth,
    constant uint& outHeight,
    constant uint& outWidth,
    uint index
) {
    uint count = batch * outChannels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    uint outCol = index % outWidth;
    uint outRow = (index / outWidth) % outHeight;
    uint outChannel = (index / (outWidth * outHeight)) % outChannels;
    uint batchIndex = index / (outWidth * outHeight * outChannels);
    float accumulator = Storage::load(bias, outChannel);

    for (uint inChannel = 0; inChannel < inChannels; inChannel++) {
        for (uint kernelRow = 0; kernelRow < kernelHeight; kernelRow++) {
            if (outRow < kernelRow) {
                continue;
            }

            uint inRow = outRow - kernelRow;

            if (inRow >= inHeight) {
                continue;
            }

            for (uint kernelCol = 0; kernelCol < kernelWidth; kernelCol++) {
                if (outCol < kernelCol) {
                    continue;
                }

                uint inCol = outCol - kernelCol;

                if (inCol >= inWidth) {
                    continue;
                }

                uint inputIndex = ((batchIndex * inChannels + inChannel) * inHeight + inRow) *
                    inWidth + inCol;
                uint weightIndex = ((inChannel * outChannels + outChannel) * kernelHeight +
                    kernelRow) * kernelWidth + kernelCol;
                accumulator += Storage::load(input, inputIndex) * Storage::load(weight, weightIndex);
            }
        }
    }

    Storage::store(out, index, accumulator);
}

template <typename Storage, typename Scalar>
static inline void pool2d_kernel(
    device const Scalar* input,
    device Scalar* out,
    constant uint& batch,
    constant uint& channels,
    constant uint& inHeight,
    constant uint& inWidth,
    constant uint& outHeight,
    constant uint& outWidth,
    constant bool& useMax,
    uint index
) {
    uint count = batch * channels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    uint outCol = index % outWidth;
    uint outRow = (index / outWidth) % outHeight;
    uint channel = (index / (outWidth * outHeight)) % channels;
    uint batchIndex = index / (outWidth * outHeight * channels);
    uint startRow = outRow * 2;
    uint startCol = outCol * 2;
    float value = useMax ? -INFINITY : 0.0f;
    uint elements = 0;

    for (uint kernelRow = 0; kernelRow < 2; kernelRow++) {
        uint inRow = startRow + kernelRow;

        if (inRow >= inHeight) {
            continue;
        }

        for (uint kernelCol = 0; kernelCol < 2; kernelCol++) {
            uint inCol = startCol + kernelCol;

            if (inCol >= inWidth) {
                continue;
            }

            uint inputIndex = ((batchIndex * channels + channel) * inHeight + inRow) *
                inWidth + inCol;
            float candidate = Storage::load(input, inputIndex);
            elements++;

            if (useMax) {
                value = candidate > value ? candidate : value;
                continue;
            }

            value += candidate;
        }
    }

    if (!useMax && elements > 0) {
        value /= float(elements);
    }

    Storage::store(out, index, value);
}

template <typename Storage, typename Scalar>
static inline void adaptive_pool2d_kernel(
    device const Scalar* input,
    device Scalar* out,
    constant uint& batch,
    constant uint& channels,
    constant uint& inHeight,
    constant uint& inWidth,
    constant uint& outHeight,
    constant uint& outWidth,
    constant bool& useMax,
    uint index
) {
    uint count = batch * channels * outHeight * outWidth;

    if (index >= count) {
        return;
    }

    uint outCol = index % outWidth;
    uint outRow = (index / outWidth) % outHeight;
    uint channel = (index / (outWidth * outHeight)) % channels;
    uint batchIndex = index / (outWidth * outHeight * channels);
    uint startRow = (outRow * inHeight) / outHeight;
    uint endRow = ((outRow + 1) * inHeight) / outHeight;
    uint startCol = (outCol * inWidth) / outWidth;
    uint endCol = ((outCol + 1) * inWidth) / outWidth;
    float value = useMax ? -1.0e30f : 0.0f;
    uint elements = 0;

    for (uint inRow = startRow; inRow < endRow; inRow++) {
        for (uint inCol = startCol; inCol < endCol; inCol++) {
            uint inputIndex = ((batchIndex * channels + channel) * inHeight + inRow) *
                inWidth + inCol;
            float candidate = Storage::load(input, inputIndex);
            elements++;

            if (useMax) {
                value = candidate > value ? candidate : value;
                continue;
            }

            value += candidate;
        }
    }

    if (!useMax && elements > 0) {
        value /= float(elements);
    }

    Storage::store(out, index, value);
}

#define CONV1D_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* weight [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& batch [[buffer(4)]], \
    constant uint& inChannels [[buffer(5)]], \
    constant uint& inLength [[buffer(6)]], \
    constant uint& outChannels [[buffer(7)]], \
    constant uint& kernelLength [[buffer(8)]], \
    constant uint& outLength [[buffer(9)]], \
    uint index [[thread_position_in_grid]] \
) { \
    conv1d_kernel<storage, scalar>( \
        input, weight, bias, out, batch, inChannels, inLength, outChannels, \
        kernelLength, outLength, index \
    ); \
}

#define CONV2D_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* weight [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& batch [[buffer(4)]], \
    constant uint& inChannels [[buffer(5)]], \
    constant uint& inHeight [[buffer(6)]], \
    constant uint& inWidth [[buffer(7)]], \
    constant uint& outChannels [[buffer(8)]], \
    constant uint& kernelHeight [[buffer(9)]], \
    constant uint& kernelWidth [[buffer(10)]], \
    constant uint& outHeight [[buffer(11)]], \
    constant uint& outWidth [[buffer(12)]], \
    uint index [[thread_position_in_grid]] \
) { \
    conv2d_kernel<storage, scalar>( \
        input, weight, bias, out, batch, inChannels, inHeight, inWidth, \
        outChannels, kernelHeight, kernelWidth, outHeight, outWidth, index \
    ); \
}

#define CONV3D_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* weight [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& batch [[buffer(4)]], \
    constant uint& inChannels [[buffer(5)]], \
    constant uint& inDepth [[buffer(6)]], \
    constant uint& inHeight [[buffer(7)]], \
    constant uint& inWidth [[buffer(8)]], \
    constant uint& outChannels [[buffer(9)]], \
    constant uint& kernelDepth [[buffer(10)]], \
    constant uint& kernelHeight [[buffer(11)]], \
    constant uint& kernelWidth [[buffer(12)]], \
    constant uint& outDepth [[buffer(13)]], \
    constant uint& outHeight [[buffer(14)]], \
    constant uint& outWidth [[buffer(15)]], \
    uint index [[thread_position_in_grid]] \
) { \
    conv3d_kernel<storage, scalar>( \
        input, weight, bias, out, batch, inChannels, inDepth, inHeight, inWidth, \
        outChannels, kernelDepth, kernelHeight, kernelWidth, outDepth, outHeight, outWidth, index \
    ); \
}

#define CONV_TRANSPOSE2D_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* weight [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& batch [[buffer(4)]], \
    constant uint& inChannels [[buffer(5)]], \
    constant uint& inHeight [[buffer(6)]], \
    constant uint& inWidth [[buffer(7)]], \
    constant uint& outChannels [[buffer(8)]], \
    constant uint& kernelHeight [[buffer(9)]], \
    constant uint& kernelWidth [[buffer(10)]], \
    constant uint& outHeight [[buffer(11)]], \
    constant uint& outWidth [[buffer(12)]], \
    uint index [[thread_position_in_grid]] \
) { \
    conv_transpose2d_kernel<storage, scalar>( \
        input, weight, bias, out, batch, inChannels, inHeight, inWidth, \
        outChannels, kernelHeight, kernelWidth, outHeight, outWidth, index \
    ); \
}

#define POOL2D_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device scalar* out [[buffer(1)]], \
    constant uint& batch [[buffer(2)]], \
    constant uint& channels [[buffer(3)]], \
    constant uint& inHeight [[buffer(4)]], \
    constant uint& inWidth [[buffer(5)]], \
    constant uint& outHeight [[buffer(6)]], \
    constant uint& outWidth [[buffer(7)]], \
    constant bool& useMax [[buffer(8)]], \
    uint index [[thread_position_in_grid]] \
) { \
    pool2d_kernel<storage, scalar>( \
        input, out, batch, channels, inHeight, inWidth, outHeight, outWidth, useMax, index \
    ); \
}

#define ADAPTIVE_POOL2D_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device scalar* out [[buffer(1)]], \
    constant uint& batch [[buffer(2)]], \
    constant uint& channels [[buffer(3)]], \
    constant uint& inHeight [[buffer(4)]], \
    constant uint& inWidth [[buffer(5)]], \
    constant uint& outHeight [[buffer(6)]], \
    constant uint& outWidth [[buffer(7)]], \
    constant bool& useMax [[buffer(8)]], \
    uint index [[thread_position_in_grid]] \
) { \
    adaptive_pool2d_kernel<storage, scalar>( \
        input, out, batch, channels, inHeight, inWidth, outHeight, outWidth, useMax, index \
    ); \
}

CONV1D_KERNEL(conv1d_float32, Float32VisionStorage, float)
CONV1D_KERNEL(conv1d_float16, Float16VisionStorage, half)
CONV1D_KERNEL(conv1d_bfloat16, BFloat16VisionStorage, ushort)

CONV2D_KERNEL(conv2d_float32, Float32VisionStorage, float)
CONV2D_KERNEL(conv2d_float16, Float16VisionStorage, half)
CONV2D_KERNEL(conv2d_bfloat16, BFloat16VisionStorage, ushort)

CONV3D_KERNEL(conv3d_float32, Float32VisionStorage, float)
CONV3D_KERNEL(conv3d_float16, Float16VisionStorage, half)
CONV3D_KERNEL(conv3d_bfloat16, BFloat16VisionStorage, ushort)

CONV_TRANSPOSE2D_KERNEL(conv_transpose2d_float32, Float32VisionStorage, float)
CONV_TRANSPOSE2D_KERNEL(conv_transpose2d_float16, Float16VisionStorage, half)
CONV_TRANSPOSE2D_KERNEL(conv_transpose2d_bfloat16, BFloat16VisionStorage, ushort)

POOL2D_KERNEL(max_pool2d_float32, Float32VisionStorage, float)
POOL2D_KERNEL(max_pool2d_float16, Float16VisionStorage, half)
POOL2D_KERNEL(max_pool2d_bfloat16, BFloat16VisionStorage, ushort)
POOL2D_KERNEL(avg_pool2d_float32, Float32VisionStorage, float)
POOL2D_KERNEL(avg_pool2d_float16, Float16VisionStorage, half)
POOL2D_KERNEL(avg_pool2d_bfloat16, BFloat16VisionStorage, ushort)

ADAPTIVE_POOL2D_KERNEL(adaptive_avg_pool2d_float32, Float32VisionStorage, float)
ADAPTIVE_POOL2D_KERNEL(adaptive_avg_pool2d_float16, Float16VisionStorage, half)
ADAPTIVE_POOL2D_KERNEL(adaptive_avg_pool2d_bfloat16, BFloat16VisionStorage, ushort)
ADAPTIVE_POOL2D_KERNEL(adaptive_max_pool2d_float32, Float32VisionStorage, float)
ADAPTIVE_POOL2D_KERNEL(adaptive_max_pool2d_float16, Float16VisionStorage, half)
ADAPTIVE_POOL2D_KERNEL(adaptive_max_pool2d_bfloat16, BFloat16VisionStorage, ushort)
