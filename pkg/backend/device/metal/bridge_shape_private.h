#ifndef CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_SHAPE_PRIVATE_H
#define CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_SHAPE_PRIVATE_H

#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

typedef void (^MetalShapeEncodeBlock)(id<MTLComputeCommandEncoder> encoder);
typedef void (^MetalShapeValidatedEncodeBlock)(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> validationBuffer
);

void metal_shape_status_set(MetalStatus* status, int code, const char* message);
int metal_shape_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
);
int metal_shape_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    MetalShapeEncodeBlock encode
);
int metal_shape_dispatch_validated(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    MetalShapeValidatedEncodeBlock encode
);

#endif
