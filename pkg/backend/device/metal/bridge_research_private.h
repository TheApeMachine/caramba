#ifndef CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_RESEARCH_PRIVATE_H
#define CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_RESEARCH_PRIVATE_H

#include "bridge_darwin_private.h"

#include <stddef.h>

void metal_research_status_set(MetalStatus* status, int code, const char* message);
int metal_research_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
);
int metal_research_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger count,
    uint64_t completionToken,
    MetalStatus* status,
    void (^encode)(id<MTLComputeCommandEncoder> encoder)
);

#endif
