#include <metal_stdlib>

using namespace metal;

constant uint samplingThreadCount = 256;
constant uint samplingInvalidIndex = 0xffffffffu;

static inline float sampling_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

struct Float32SamplingStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }
};

struct Float16SamplingStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }
};

struct BFloat16SamplingStorage {
    static float load(device const ushort* values, uint index) {
        return sampling_bf16_to_float(values[index]);
    }
};

static inline bool sampling_candidate_before(
    float leftScore,
    uint leftIndex,
    float rightScore,
    uint rightIndex
) {
    if (leftScore > rightScore) {
        return true;
    }

    if (leftScore < rightScore) {
        return false;
    }

    return leftIndex < rightIndex;
}

template <typename Storage, typename Scalar>
static inline void greedy_sample_kernel(
    device const Scalar* logits,
    device int* out,
    threadgroup float* scoreScratch,
    threadgroup uint* indexScratch,
    constant uint& count,
    uint threadIndex
) {
    float bestScore = -3.4028234663852886e38f;
    uint bestIndex = samplingInvalidIndex;

    for (uint index = threadIndex; index < count; index += samplingThreadCount) {
        float score = Storage::load(logits, index);
        if (sampling_candidate_before(score, index, bestScore, bestIndex)) {
            bestScore = score;
            bestIndex = index;
        }
    }

    scoreScratch[threadIndex] = bestScore;
    indexScratch[threadIndex] = bestIndex;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = samplingThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            float rightScore = scoreScratch[threadIndex + stride];
            uint rightIndex = indexScratch[threadIndex + stride];
            if (sampling_candidate_before(rightScore, rightIndex, scoreScratch[threadIndex], indexScratch[threadIndex])) {
                scoreScratch[threadIndex] = rightScore;
                indexScratch[threadIndex] = rightIndex;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        out[0] = int(indexScratch[0]);
    }
}

template <typename Storage, typename Scalar>
static inline void sampling_init_kernel(
    device const Scalar* logits,
    device float* scores,
    device uint* indices,
    constant uint& count,
    constant uint& paddedCount,
    uint index
) {
    if (index >= paddedCount) {
        return;
    }

    if (index < count) {
        scores[index] = Storage::load(logits, index);
        indices[index] = index;
        return;
    }

    scores[index] = -3.4028234663852886e38f;
    indices[index] = samplingInvalidIndex;
}

kernel void sampling_bitonic_step(
    device float* scores [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant uint& stageSize [[buffer(2)]],
    constant uint& passSize [[buffer(3)]],
    constant uint& paddedCount [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= paddedCount) {
        return;
    }

    uint peer = index ^ passSize;
    if (peer <= index || peer >= paddedCount) {
        return;
    }

    float leftScore = scores[index];
    uint leftIndex = indices[index];
    float rightScore = scores[peer];
    uint rightIndex = indices[peer];
    bool descending = (index & stageSize) == 0;
    bool rightBeforeLeft = sampling_candidate_before(rightScore, rightIndex, leftScore, leftIndex);
    bool leftBeforeRight = sampling_candidate_before(leftScore, leftIndex, rightScore, rightIndex);

    if (descending && !rightBeforeLeft) {
        return;
    }

    if (!descending && !leftBeforeRight) {
        return;
    }

    scores[index] = rightScore;
    indices[index] = rightIndex;
    scores[peer] = leftScore;
    indices[peer] = leftIndex;
}

kernel void sampling_draw_sorted(
    device const float* scores [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device int* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& target [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index != 0) {
        return;
    }

    float maximum = scores[0];
    float denominator = 0.0f;

    for (uint scoreIndex = 0; scoreIndex < count; scoreIndex++) {
        denominator += exp(scores[scoreIndex] - maximum);
    }

    float targetMass = target * denominator;
    float cumulative = 0.0f;
    uint chosen = indices[count - 1];

    for (uint scoreIndex = 0; scoreIndex < count; scoreIndex++) {
        cumulative += exp(scores[scoreIndex] - maximum);

        if (cumulative >= targetMass) {
            chosen = indices[scoreIndex];
            break;
        }
    }

    out[0] = int(chosen);
}

#define GREEDY_SAMPLE_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* logits [[buffer(0)]], \
    device int* out [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float scoreScratch[256]; \
    threadgroup uint indexScratch[256]; \
    greedy_sample_kernel<storage, scalar>(logits, out, scoreScratch, indexScratch, count, threadIndex); \
}

#define SAMPLING_INIT_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* logits [[buffer(0)]], \
    device float* scores [[buffer(1)]], \
    device uint* indices [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    constant uint& paddedCount [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    sampling_init_kernel<storage, scalar>(logits, scores, indices, count, paddedCount, index); \
}

GREEDY_SAMPLE_KERNEL(greedy_sample_float32, Float32SamplingStorage, float)
GREEDY_SAMPLE_KERNEL(greedy_sample_float16, Float16SamplingStorage, half)
GREEDY_SAMPLE_KERNEL(greedy_sample_bfloat16, BFloat16SamplingStorage, ushort)

SAMPLING_INIT_KERNEL(sampling_init_float32, Float32SamplingStorage, float)
SAMPLING_INIT_KERNEL(sampling_init_float16, Float16SamplingStorage, half)
SAMPLING_INIT_KERNEL(sampling_init_bfloat16, BFloat16SamplingStorage, ushort)
