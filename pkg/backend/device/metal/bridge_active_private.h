#ifndef CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_ACTIVE_PRIVATE_H
#define CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_ACTIVE_PRIVATE_H

#include "bridge_darwin_private.h"

#include <stddef.h>

void metal_active_status_set(MetalStatus* status, int code, const char* message);
int metal_active_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* phase,
    int elementDType,
    MetalStatus* status
);
int metal_active_prepare(
    MetalDeviceRef contextRef,
    MetalStatus* status,
    MetalContext** context,
    id<MTLCommandBuffer>* commandBuffer
);
int metal_active_pipeline(
    MetalContext* context,
    const char* kernelName,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
);
int metal_active_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    id<MTLComputeCommandEncoder>* encoder
);
void metal_active_complete(uint64_t completionToken, id<MTLCommandBuffer> completedBuffer);
int metal_active_encode_finalize(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t partialCount,
    MetalStatus* status
);

#endif
