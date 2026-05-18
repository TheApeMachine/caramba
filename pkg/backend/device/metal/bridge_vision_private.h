#ifndef CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_VISION_PRIVATE_H
#define CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_VISION_PRIVATE_H

#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

typedef void (^MetalVisionEncodeBlock)(id<MTLComputeCommandEncoder> encoder);

void metal_vision_status_set(MetalStatus* status, int code, const char* message);

int metal_vision_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
);

int metal_vision_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    MetalVisionEncodeBlock encode
);

#endif
