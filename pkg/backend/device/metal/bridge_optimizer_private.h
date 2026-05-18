#ifndef CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_OPTIMIZER_PRIVATE_H
#define CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_OPTIMIZER_PRIVATE_H

#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

typedef void (^MetalOptimizerEncodeBlock)(id<MTLComputeCommandEncoder> encoder);

void metal_optimizer_status_set(
    MetalStatus* status,
    int code,
    const char* message
);

int metal_optimizer_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
);

int metal_optimizer_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandQueue>* queue,
    id<MTLComputePipelineState>* pipeline
);

void metal_optimizer_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer
);

int metal_optimizer_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    MetalOptimizerEncodeBlock encode
);

#endif
