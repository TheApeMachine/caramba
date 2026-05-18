#ifndef CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_CAUSAL_PRIVATE_H
#define CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_CAUSAL_PRIVATE_H

#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

typedef void (^MetalCausalEncodeBlock)(id<MTLComputeCommandEncoder> encoder);

void metal_causal_status_set(MetalStatus* status, int code, const char* message);
int metal_causal_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
);
int metal_causal_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    MetalCausalEncodeBlock encode
);
int metal_causal_prepare(
    MetalDeviceRef contextRef,
    MetalStatus* status,
    MetalContext** context,
    id<MTLCommandBuffer>* commandBuffer
);
int metal_causal_pipeline(
    MetalContext* context,
    const char* kernelName,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
);
int metal_causal_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    id<MTLComputeCommandEncoder>* encoder
);
void metal_causal_complete(uint64_t completionToken, id<MTLCommandBuffer> completedBuffer);

#endif
