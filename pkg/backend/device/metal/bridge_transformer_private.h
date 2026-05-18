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

#endif
