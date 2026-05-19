#ifndef CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_SAMPLING_PRIVATE_H
#define CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_SAMPLING_PRIVATE_H

#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

void metal_sampling_status_clear(MetalStatus* status);
void metal_sampling_status_set(MetalStatus* status, int code, const char* message);
int metal_sampling_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
);
void metal_sampling_complete(uint64_t completionToken, id<MTLCommandBuffer> completedBuffer);
int metal_sampling_prepare(
    MetalDeviceRef contextRef,
    MetalStatus* status,
    MetalContext** context,
    id<MTLCommandBuffer>* commandBuffer
);
int metal_sampling_pipeline(
    MetalContext* context,
    const char* kernelName,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
);
int metal_sampling_encode_greedy(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef logitsRef,
    MetalBufferRef outRef,
    uint32_t count,
    MetalStatus* status
);
int metal_sampling_encode_init(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef logitsRef,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    uint32_t count,
    uint32_t paddedCount,
    MetalStatus* status
);
int metal_sampling_encode_bitonic(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    uint32_t stageSize,
    uint32_t passSize,
    uint32_t paddedCount,
    MetalStatus* status
);
int metal_sampling_encode_draw(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t count,
    float target,
    MetalStatus* status
);
int metal_sampling_encode_sort(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    uint32_t paddedCount,
    MetalStatus* status
);

#endif
