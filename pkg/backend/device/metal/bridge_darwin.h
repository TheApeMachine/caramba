#ifndef CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_DARWIN_H
#define CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_DARWIN_H

#include <stdint.h>
#include <stdbool.h>

#define METAL_STATUS_MESSAGE_BYTES 1024

#ifdef __cplusplus
extern "C" {
#endif

typedef void* MetalDeviceRef;
typedef void* MetalBufferRef;

typedef struct MetalStatus {
    int code;
    char message[METAL_STATUS_MESSAGE_BYTES];
} MetalStatus;

typedef enum MetalBinaryFloat32Op {
    MetalBinaryFloat32Add = 0,
    MetalBinaryFloat32Sub = 1,
    MetalBinaryFloat32Mul = 2,
    MetalBinaryFloat32Div = 3,
    MetalBinaryFloat32Max = 4,
    MetalBinaryFloat32Min = 5,
    MetalBinaryFloat32Eq = 6,
    MetalBinaryFloat32Ne = 7,
    MetalBinaryFloat32Lt = 8,
    MetalBinaryFloat32Le = 9,
    MetalBinaryFloat32Gt = 10,
    MetalBinaryFloat32Ge = 11,
    MetalBinaryFloat32Pow = 12,
    MetalBinaryFloat32Atan2 = 13,
    MetalBinaryFloat32Mod = 14,
} MetalBinaryFloat32Op;

typedef enum MetalUnaryFloat32Op {
    MetalUnaryFloat32Relu = 0,
    MetalUnaryFloat32Abs = 1,
    MetalUnaryFloat32Neg = 2,
    MetalUnaryFloat32Square = 3,
    MetalUnaryFloat32Recip = 4,
    MetalUnaryFloat32Sqrt = 5,
    MetalUnaryFloat32Sign = 6,
    MetalUnaryFloat32Rsqrt = 7,
    MetalUnaryFloat32Exp = 8,
    MetalUnaryFloat32Log = 9,
    MetalUnaryFloat32Sin = 10,
    MetalUnaryFloat32Cos = 11,
    MetalUnaryFloat32Tanh = 12,
    MetalUnaryFloat32Sigmoid = 13,
    MetalUnaryFloat32Silu = 14,
    MetalUnaryFloat32Swish = 15,
    MetalUnaryFloat32Softsign = 16,
    MetalUnaryFloat32ELU = 17,
    MetalUnaryFloat32SELU = 18,
    MetalUnaryFloat32LeakyReLU = 19,
    MetalUnaryFloat32HardSigmoid = 20,
    MetalUnaryFloat32HardSwish = 21,
} MetalUnaryFloat32Op;

typedef enum MetalElementDType {
    MetalElementDTypeFloat32 = 0,
    MetalElementDTypeFloat16 = 1,
    MetalElementDTypeBFloat16 = 2,
} MetalElementDType;

MetalDeviceRef metal_open_default_device(
    const uint8_t* libraryBytes,
    long long libraryLength,
    MetalStatus* status
);
long long metal_recommended_max_working_set(MetalDeviceRef contextRef);
MetalBufferRef metal_buffer_new_shared(MetalDeviceRef contextRef, long long bytes);
void metal_buffer_release(MetalBufferRef bufferRef);
void* metal_buffer_contents(MetalBufferRef bufferRef);
int metal_dispatch_binary_float32(
    MetalDeviceRef contextRef,
    int operation,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_unary_float32(
    MetalDeviceRef contextRef,
    int operation,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_binary_elementwise(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_unary_elementwise(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_copy_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t byteCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_concat_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t leftBytes,
    uint32_t rightBytes,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_split2_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    uint32_t leftBytes,
    uint32_t rightBytes,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_last_token_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t seq,
    uint32_t hiddenBytes,
    uint32_t outBytes,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_transpose2d_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_upsample_nearest2d_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t channels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint32_t outElements,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_gather(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef sourceRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t sourceRows,
    uint32_t inner,
    uint32_t outRows,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_scatter(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef targetRef,
    MetalBufferRef indicesRef,
    MetalBufferRef updatesRef,
    MetalBufferRef outRef,
    uint32_t targetRows,
    uint32_t inner,
    uint32_t updateRows,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_where(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef maskRef,
    MetalBufferRef positiveRef,
    MetalBufferRef negativeRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_masked_fill(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef maskRef,
    MetalBufferRef scalarRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_transpose(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rank,
    uint32_t count,
    const uint32_t* permutation,
    const uint32_t* inputStrides,
    const uint32_t* outStrides,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_matmul(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t inner,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_matmul_add(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t inner,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_softmax(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_layernorm(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scaleRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_rmsnorm(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scaleRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_groupnorm(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scaleRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial,
    uint32_t groups,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_instancenorm(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scaleRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_batchnorm_eval(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scaleRef,
    MetalBufferRef biasRef,
    MetalBufferRef meanRef,
    MetalBufferRef varianceRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_linear(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef weightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t inner,
    uint32_t outDim,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_fused_qkv(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef weightRef,
    MetalBufferRef biasRef,
    MetalBufferRef queryRef,
    MetalBufferRef keyRef,
    MetalBufferRef valueRef,
    uint32_t batch,
    uint32_t inner,
    uint32_t outDim,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_lora_merge(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef baseRef,
    MetalBufferRef loraARef,
    MetalBufferRef loraBRef,
    MetalBufferRef outRef,
    uint32_t outDim,
    uint32_t rank,
    uint32_t inner,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_lora_apply(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef baseRef,
    MetalBufferRef loraARef,
    MetalBufferRef loraBRef,
    MetalBufferRef inputRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t inner,
    uint32_t rank,
    uint32_t outDim,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_embedding_lookup(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef tableRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t vocab,
    uint32_t hidden,
    uint32_t indexCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_embedding_bag(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef tableRef,
    MetalBufferRef indicesRef,
    MetalBufferRef offsetsRef,
    MetalBufferRef outRef,
    uint32_t vocab,
    uint32_t hidden,
    uint32_t indexCount,
    uint32_t bagCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_attention(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef queryRef,
    MetalBufferRef keyRef,
    MetalBufferRef valueRef,
    MetalBufferRef scoresRef,
    MetalBufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t depth,
    uint32_t valueDim,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_flash_attention(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef queryRef,
    MetalBufferRef keyRef,
    MetalBufferRef valueRef,
    MetalBufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t depth,
    uint32_t valueDim,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_multi_head_attention(
    MetalDeviceRef contextRef,
    int elementDType,
    int variant,
    MetalBufferRef queryRef,
    MetalBufferRef keyRef,
    MetalBufferRef valueRef,
    MetalBufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t numHeads,
    uint32_t kvHeads,
    uint32_t headDim,
    uint32_t windowSize,
    uint32_t causal,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_rope(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t seqLen,
    uint32_t numHeads,
    uint32_t headDim,
    uint32_t pairCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_apply_mask(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef maskRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_causal_mask(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_alibi_bias(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef scoresRef,
    MetalBufferRef slopeRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_backdoor_adjustment(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef conditionalRef,
    MetalBufferRef marginalRef,
    MetalBufferRef outRef,
    uint32_t xCount,
    uint32_t zCount,
    uint32_t yCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_frontdoor_adjustment(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef mediatorRef,
    MetalBufferRef outcomeRef,
    MetalBufferRef marginalRef,
    MetalBufferRef outRef,
    uint32_t xCount,
    uint32_t mCount,
    uint32_t yCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_do_intervene(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef adjacencyRef,
    MetalBufferRef intervenedRef,
    MetalBufferRef outRef,
    uint32_t nodeCount,
    uint32_t intervenedCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_cate(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef treatedRef,
    MetalBufferRef controlRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_counterfactual(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef observedYRef,
    MetalBufferRef observedXRef,
    MetalBufferRef counterfactualXRef,
    MetalBufferRef slopeRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_iv_estimate(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef instrumentRef,
    MetalBufferRef treatmentRef,
    MetalBufferRef outcomeRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_dag_markov_factorization(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef conditionalsRef,
    MetalBufferRef parentsRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_conv1d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef weightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t inChannels,
    uint32_t inLength,
    uint32_t outChannels,
    uint32_t kernelLength,
    uint32_t outLength,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_conv2d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef weightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t inChannels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outChannels,
    uint32_t kernelHeight,
    uint32_t kernelWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_conv3d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef weightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t inChannels,
    uint32_t inDepth,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outChannels,
    uint32_t kernelDepth,
    uint32_t kernelHeight,
    uint32_t kernelWidth,
    uint32_t outDepth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_conv_transpose2d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef weightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t inChannels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outChannels,
    uint32_t kernelHeight,
    uint32_t kernelWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_pool2d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    bool useMax,
    bool adaptive,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_optimizer4(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef firstRef,
    MetalBufferRef secondRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_optimizer3(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef stateRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_optimizer2(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_hebbian_step(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef weightsRef,
    MetalBufferRef postRef,
    MetalBufferRef preRef,
    MetalBufferRef outRef,
    uint32_t postCount,
    uint32_t preCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_lars_step(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef momentumRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t groupCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_quantization(
    MetalDeviceRef contextRef,
    int operation,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_pair_loss(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef predictionsRef,
    MetalBufferRef targetsRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_cross_entropy_loss(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef logitsRef,
    MetalBufferRef targetsRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t classes,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_reduction(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scratchARef,
    MetalBufferRef scratchBRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_inv_sqrt_dim_scale(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef dimRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_logsumexp(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_outer(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_sampling(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef logitsRef,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t paddedCount,
    float target,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_dropout(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    float scale,
    uint32_t threshold,
    uint32_t seed0,
    uint32_t seed1,
    uint32_t seed2,
    uint32_t seed3,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_research_unary(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_research_binary(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_pc_prediction(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef weightsRef,
    MetalBufferRef stateRef,
    MetalBufferRef outRef,
    uint32_t outCount,
    uint32_t inCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_pc_update_representation(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef weightsRef,
    MetalBufferRef stateRef,
    MetalBufferRef errorRef,
    MetalBufferRef outRef,
    uint32_t outCount,
    uint32_t inCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_pc_update_weights(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef weightsRef,
    MetalBufferRef stateRef,
    MetalBufferRef errorRef,
    MetalBufferRef outRef,
    uint32_t outCount,
    uint32_t inCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_active_free_energy(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef likelihoodRef,
    MetalBufferRef posteriorRef,
    MetalBufferRef priorRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_expected_free_energy(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef predictedObsRef,
    MetalBufferRef preferredObsRef,
    MetalBufferRef predictedStateRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t obsCount,
    uint32_t stateCount,
    uint32_t obsPartialCount,
    uint32_t statePartialCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_belief_update(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef likelihoodRef,
    MetalBufferRef priorRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_precision_weight(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef errorsRef,
    MetalBufferRef precisionRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_hawkes_intensity(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef eventsRef,
    MetalBufferRef queryTimesRef,
    MetalBufferRef baselineRef,
    MetalBufferRef alphaRef,
    MetalBufferRef betaRef,
    MetalBufferRef outRef,
    uint32_t eventCount,
    uint32_t queryCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_hawkes_kernel_matrix(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef eventsRef,
    MetalBufferRef alphaRef,
    MetalBufferRef betaRef,
    MetalBufferRef outRef,
    uint32_t eventCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_hawkes_log_likelihood(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef eventsRef,
    MetalBufferRef totalTimeRef,
    MetalBufferRef baselineRef,
    MetalBufferRef alphaRef,
    MetalBufferRef betaRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t eventCount,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_markov_mutual_information(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef jointRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_markov_blanket_partition(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef adjacencyRef,
    MetalBufferRef internalRef,
    MetalBufferRef outRef,
    uint32_t nodeCount,
    uint32_t internalCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_markov_flow(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef mutualInformationRef,
    MetalBufferRef partitionRef,
    MetalBufferRef outRef,
    uint32_t nodeCount,
    int32_t targetLabel,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_laplacian(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef spacingRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t rank,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_laplacian4(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef spacingRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_grad1d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef spacingRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_divergence1d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef spacingRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_quantum_potential(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef densityRef,
    MetalBufferRef spacingRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_bohmian_velocity(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef phaseRef,
    MetalBufferRef spacingRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_madelung_continuity(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef densityRef,
    MetalBufferRef velocityRef,
    MetalBufferRef spacingRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_fft1d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef realInRef,
    MetalBufferRef imagInRef,
    MetalBufferRef realOutRef,
    MetalBufferRef imagOutRef,
    uint32_t count,
    bool inverse,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_checkpoint_encode_float32(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rank,
    uint32_t count,
    const uint64_t* dims,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_checkpoint_decode_float32(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t headerBytes,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_tokenizer_pack_int32(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_weight_freeze_mask(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef maskRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
void metal_device_release(MetalDeviceRef contextRef);

#ifdef __cplusplus
}
#endif

#endif
