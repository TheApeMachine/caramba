#include <metal_stdlib>

using namespace metal;

constant uint normalizationThreadCount = 256;
constant float layerNormEpsilonMetal = 1.0e-5f;
constant float rmsNormEpsilonMetal = 1.0e-6f;

static inline float bf16_to_float_norm(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort float_to_bf16_norm(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline float refined_inv_sqrt_norm(float value) {
    float estimate = 1.0f / precise::sqrt(value);
    float halfValue = 0.5f * value;
    estimate = estimate * (1.5f - halfValue * estimate * estimate);
    return estimate;
}

static inline float2 ds_renorm(float high, float low) {
    float sum = high + low;
    float error = low - (sum - high);
    return float2(sum, error);
}

static inline float2 ds_add_float(float2 value, float addend) {
    float sum = value.x + addend;
    float virtualAddend = sum - value.x;
    float error = (value.x - (sum - virtualAddend)) + (addend - virtualAddend);
    return ds_renorm(sum, value.y + error);
}

static inline float2 ds_add_pair(float2 left, float2 right) {
    float2 withHigh = ds_add_float(left, right.x);
    return ds_add_float(withHigh, right.y);
}

static inline float2 ds_neg(float2 value) {
    return float2(-value.x, -value.y);
}

static inline float2 ds_sub_from_float(float value, float2 subtrahend) {
    return ds_add_float(ds_neg(subtrahend), value);
}

static inline float2 ds_mul_pair(float2 left, float2 right) {
    float product = left.x * right.x;
    float error = fma(left.x, right.x, -product) + left.x * right.y + left.y * right.x;
    return ds_renorm(product, error);
}

static inline float2 ds_mul_float(float2 value, float scalar) {
    float product = value.x * scalar;
    float error = fma(value.x, scalar, -product) + value.y * scalar;
    return ds_renorm(product, error);
}

static inline float2 ds_div_count(float2 value, uint count) {
    return ds_mul_float(value, 1.0f / float(count));
}

static inline float ds_to_float(float2 value) {
    return value.x + value.y;
}

static inline float ds_inv_sqrt(float2 value, float epsilon) {
    float high = value.x + epsilon;
    float estimate = refined_inv_sqrt_norm(high);
    return estimate * (1.0f - 0.5f * value.y / high);
}

struct Float32NormStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16NormStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16NormStorage {
    static float load(device const ushort* values, uint index) {
        return bf16_to_float_norm(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = float_to_bf16_norm(value);
    }
};

template <typename Storage, typename Scalar>
static inline float reduce_sum(
    device const Scalar* input,
    threadgroup float* reduction,
    uint rowOffset,
    uint cols,
    uint threadIndex
) {
    float localSum = 0.0f;
    float localCompensation = 0.0f;

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float value = Storage::load(input, rowOffset + col) - localCompensation;
        float nextSum = localSum + value;
        localCompensation = (nextSum - localSum) - value;
        localSum = nextSum;
    }

    reduction[threadIndex] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    return reduction[0];
}

template <typename Storage, typename Scalar>
static inline void layernorm_rows(
    device const Scalar* input,
    device const Scalar* scale,
    device const Scalar* bias,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& cols,
    uint row,
    uint threadIndex
) {
    uint rowOffset = row * cols;
    float mean = reduce_sum<Storage, Scalar>(input, reduction, rowOffset, cols, threadIndex) /
        float(cols);
    float localVariance = 0.0f;
    float localCompensation = 0.0f;

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float delta = Storage::load(input, rowOffset + col) - mean;
        float value = delta * delta - localCompensation;
        float nextVariance = localVariance + value;
        localCompensation = (nextVariance - localVariance) - value;
        localVariance = nextVariance;
    }

    reduction[threadIndex] = localVariance;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float invStdDev = 1.0f / sqrt(reduction[0] / float(cols) + layerNormEpsilonMetal);

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float normalized = (Storage::load(input, rowOffset + col) - mean) * invStdDev;
        float value = normalized * Storage::load(scale, col) + Storage::load(bias, col);
        Storage::store(out, rowOffset + col, value);
    }
}

template <typename Storage, typename Scalar>
static inline void rmsnorm_rows(
    device const Scalar* input,
    device const Scalar* scale,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& cols,
    uint row,
    uint threadIndex
) {
    uint rowOffset = row * cols;
    float localSquareSum = 0.0f;
    float localCompensation = 0.0f;

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float value = Storage::load(input, rowOffset + col);
        float square = value * value - localCompensation;
        float nextSum = localSquareSum + square;
        localCompensation = (nextSum - localSquareSum) - square;
        localSquareSum = nextSum;
    }

    reduction[threadIndex] = localSquareSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float invRMS = 1.0f / sqrt(reduction[0] / float(cols) + rmsNormEpsilonMetal);

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float value = Storage::load(input, rowOffset + col) * invRMS * Storage::load(scale, col);
        Storage::store(out, rowOffset + col, value);
    }
}

template <typename Storage, typename Scalar>
static inline void groupnorm_rows(
    device const Scalar* input,
    device const Scalar* scale,
    device const Scalar* bias,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& channels,
    constant uint& spatial,
    constant uint& groups,
    uint row,
    uint threadIndex
) {
    uint groupIndex = row % groups;
    uint batchIndex = row / groups;
    uint channelsPerGroup = channels / groups;
    uint channelStart = groupIndex * channelsPerGroup;
    uint groupSize = channelsPerGroup * spatial;
    uint groupOffset = (batchIndex * channels + channelStart) * spatial;
    float mean = reduce_sum<Storage, Scalar>(input, reduction, groupOffset, groupSize, threadIndex) /
        float(groupSize);
    float localVariance = 0.0f;

    for (uint offset = threadIndex; offset < groupSize; offset += normalizationThreadCount) {
        float delta = Storage::load(input, groupOffset + offset) - mean;
        localVariance += delta * delta;
    }

    reduction[threadIndex] = localVariance;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float invStdDev = 1.0f / sqrt(reduction[0] / float(groupSize) + layerNormEpsilonMetal);

    for (uint offset = threadIndex; offset < groupSize; offset += normalizationThreadCount) {
        uint channel = channelStart + offset / spatial;
        float normalized = (Storage::load(input, groupOffset + offset) - mean) * invStdDev;
        float value = normalized * Storage::load(scale, channel) + Storage::load(bias, channel);
        Storage::store(out, groupOffset + offset, value);
    }
}

template <typename Storage, typename Scalar>
static inline void instancenorm_rows(
    device const Scalar* input,
    device const Scalar* scale,
    device const Scalar* bias,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& channels,
    constant uint& spatial,
    uint row,
    uint threadIndex
) {
    uint channel = row % channels;
    uint rowOffset = row * spatial;
    float mean = reduce_sum<Storage, Scalar>(input, reduction, rowOffset, spatial, threadIndex) /
        float(spatial);
    float localVariance = 0.0f;

    for (uint offset = threadIndex; offset < spatial; offset += normalizationThreadCount) {
        float delta = Storage::load(input, rowOffset + offset) - mean;
        localVariance += delta * delta;
    }

    reduction[threadIndex] = localVariance;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float invStdDev = 1.0f / sqrt(reduction[0] / float(spatial) + layerNormEpsilonMetal);

    for (uint offset = threadIndex; offset < spatial; offset += normalizationThreadCount) {
        float normalized = (Storage::load(input, rowOffset + offset) - mean) * invStdDev;
        float value = normalized * Storage::load(scale, channel) + Storage::load(bias, channel);
        Storage::store(out, rowOffset + offset, value);
    }
}

template <typename Storage, typename Scalar>
static inline void batchnorm_eval_rows(
    device const Scalar* input,
    device const Scalar* scale,
    device const Scalar* bias,
    device const Scalar* mean,
    device const Scalar* variance,
    device Scalar* out,
    constant uint& channels,
    constant uint& spatial,
    uint row,
    uint threadIndex
) {
    uint channel = row % channels;
    uint rowOffset = row * spatial;
    float invStdDev = 1.0f / sqrt(Storage::load(variance, channel) + layerNormEpsilonMetal);
    float channelMean = Storage::load(mean, channel);
    float channelScale = Storage::load(scale, channel);
    float channelBias = Storage::load(bias, channel);

    for (uint offset = threadIndex; offset < spatial; offset += normalizationThreadCount) {
        float normalized = (Storage::load(input, rowOffset + offset) - channelMean) * invStdDev;
        Storage::store(out, rowOffset + offset, normalized * channelScale + channelBias);
    }
}

#define LAYERNORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* scale [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& cols [[buffer(4)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    layernorm_rows<storage, scalar>(input, scale, bias, out, reduction, cols, row, threadIndex); \
}

#define RMSNORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* scale [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& cols [[buffer(3)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    rmsnorm_rows<storage, scalar>(input, scale, out, reduction, cols, row, threadIndex); \
}

#define GROUPNORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* scale [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& channels [[buffer(4)]], \
    constant uint& spatial [[buffer(5)]], \
    constant uint& groups [[buffer(6)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    groupnorm_rows<storage, scalar>( \
        input, scale, bias, out, reduction, channels, spatial, groups, row, threadIndex \
    ); \
}

#define INSTANCENORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* scale [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& channels [[buffer(4)]], \
    constant uint& spatial [[buffer(5)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    instancenorm_rows<storage, scalar>( \
        input, scale, bias, out, reduction, channels, spatial, row, threadIndex \
    ); \
}

#define BATCHNORM_EVAL_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* scale [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device const scalar* mean [[buffer(3)]], \
    device const scalar* variance [[buffer(4)]], \
    device scalar* out [[buffer(5)]], \
    constant uint& channels [[buffer(6)]], \
    constant uint& spatial [[buffer(7)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    batchnorm_eval_rows<storage, scalar>( \
        input, scale, bias, mean, variance, out, channels, spatial, row, threadIndex \
    ); \
}

kernel void layernorm_float32(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    threadgroup float stats[2];
    uint rowOffset = row * cols;

    if (threadIndex == 0) {
        float sum = 0.0f;

        for (uint col = 0; col < cols; col++) {
            sum += input[rowOffset + col];
        }

        float mean = sum / float(cols);
        float variance = 0.0f;

        for (uint col = 0; col < cols; col++) {
            float delta = input[rowOffset + col] - mean;
            variance += delta * delta;
        }

        stats[0] = mean;
        stats[1] = 1.0f / precise::sqrt(variance / float(cols) + layerNormEpsilonMetal);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = stats[0];
    float invStdDev = stats[1];

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float normalized = (input[rowOffset + col] - mean) * invStdDev;
        out[rowOffset + col] = normalized * scale[col] + bias[col];
    }
}

LAYERNORM_KERNEL(layernorm_float16, Float16NormStorage, half)
LAYERNORM_KERNEL(layernorm_bfloat16, BFloat16NormStorage, ushort)

RMSNORM_KERNEL(rmsnorm_float32, Float32NormStorage, float)
RMSNORM_KERNEL(rmsnorm_float16, Float16NormStorage, half)
RMSNORM_KERNEL(rmsnorm_bfloat16, BFloat16NormStorage, ushort)

kernel void groupnorm_float32(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& spatial [[buffer(5)]],
    constant uint& groups [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    threadgroup float stats[3];
    uint groupIndex = row % groups;
    uint batchIndex = row / groups;
    uint channelsPerGroup = channels / groups;
    uint channelStart = groupIndex * channelsPerGroup;
    uint groupSize = channelsPerGroup * spatial;
    uint groupOffset = (batchIndex * channels + channelStart) * spatial;

    if (threadIndex == 0) {
        float2 sum = float2(0.0f, 0.0f);

        for (uint offset = 0; offset < groupSize; offset++) {
            sum = ds_add_float(sum, input[groupOffset + offset]);
        }

        float2 mean = ds_div_count(sum, groupSize);
        float2 variance = float2(0.0f, 0.0f);

        for (uint offset = 0; offset < groupSize; offset++) {
            float2 delta = ds_sub_from_float(input[groupOffset + offset], mean);
            variance = ds_add_pair(variance, ds_mul_pair(delta, delta));
        }

        float2 varianceMean = ds_div_count(variance, groupSize);
        stats[0] = mean.x;
        stats[1] = mean.y;
        stats[2] = ds_inv_sqrt(varianceMean, layerNormEpsilonMetal);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 mean = float2(stats[0], stats[1]);
    float invStdDev = stats[2];

    for (uint offset = threadIndex; offset < groupSize; offset += normalizationThreadCount) {
        uint channel = channelStart + offset / spatial;
        float2 centered = ds_sub_from_float(input[groupOffset + offset], mean);
        float normalized = ds_to_float(ds_mul_float(centered, invStdDev));
        out[groupOffset + offset] = fma(normalized, scale[channel], bias[channel]);
    }
}

GROUPNORM_KERNEL(groupnorm_float16, Float16NormStorage, half)
GROUPNORM_KERNEL(groupnorm_bfloat16, BFloat16NormStorage, ushort)

kernel void instancenorm_float32(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& spatial [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    threadgroup float stats[3];
    uint channel = row % channels;
    uint rowOffset = row * spatial;

    if (threadIndex == 0) {
        float2 sum = float2(0.0f, 0.0f);

        for (uint offset = 0; offset < spatial; offset++) {
            sum = ds_add_float(sum, input[rowOffset + offset]);
        }

        float2 mean = ds_div_count(sum, spatial);
        float2 variance = float2(0.0f, 0.0f);

        for (uint offset = 0; offset < spatial; offset++) {
            float2 delta = ds_sub_from_float(input[rowOffset + offset], mean);
            variance = ds_add_pair(variance, ds_mul_pair(delta, delta));
        }

        float2 varianceMean = ds_div_count(variance, spatial);
        stats[0] = mean.x;
        stats[1] = mean.y;
        stats[2] = ds_inv_sqrt(varianceMean, layerNormEpsilonMetal);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 mean = float2(stats[0], stats[1]);
    float invStdDev = stats[2];
    float channelScale = scale[channel];
    float channelBias = bias[channel];

    for (uint offset = threadIndex; offset < spatial; offset += normalizationThreadCount) {
        float2 centered = ds_sub_from_float(input[rowOffset + offset], mean);
        float normalized = ds_to_float(ds_mul_float(centered, invStdDev));
        out[rowOffset + offset] = fma(normalized, channelScale, channelBias);
    }
}

INSTANCENORM_KERNEL(instancenorm_float16, Float16NormStorage, half)
INSTANCENORM_KERNEL(instancenorm_bfloat16, BFloat16NormStorage, ushort)

BATCHNORM_EVAL_KERNEL(batchnorm_eval_float32, Float32NormStorage, float)
BATCHNORM_EVAL_KERNEL(batchnorm_eval_float16, Float16NormStorage, half)
BATCHNORM_EVAL_KERNEL(batchnorm_eval_bfloat16, BFloat16NormStorage, ushort)
