#ifndef CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_TRANSFORMER_PRIVATE_H
#define CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_TRANSFORMER_PRIVATE_H

#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

typedef void (^MetalTransformerEncodeBlock)(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> validationBuffer
);

void metal_transformer_status_set(
    MetalStatus* status,
    int code,
    const char* message
);

int metal_transformer_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
);

int metal_transformer_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    bool needsValidation,
    uint64_t completionToken,
    MetalStatus* status,
    MetalTransformerEncodeBlock encode
);

void metal_attention_status_clear(MetalStatus* status);

void metal_attention_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer
);

int metal_attention_pipeline(
    MetalContext* context,
    const char* operationName,
    int elementDType,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
);

id<MTLCommandBuffer> metal_attention_command_buffer(
    MetalContext* context,
    MetalStatus* status
);

id<MTLComputeCommandEncoder> metal_attention_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status
);

#endif
