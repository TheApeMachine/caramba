#ifndef CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_HAWKES_MARKOV_PRIVATE_H
#define CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_HAWKES_MARKOV_PRIVATE_H

#include "bridge_darwin_private.h"

#include <stddef.h>

void metal_hm_status_set(MetalStatus* status, int code, const char* message);
int metal_hm_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
);
int metal_hm_phase_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* phase,
    int elementDType,
    MetalStatus* status
);
int metal_hm_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandBuffer>* commandBuffer,
    id<MTLComputePipelineState>* pipeline
);
int metal_hm_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    id<MTLComputeCommandEncoder>* encoder
);
void metal_hm_complete(uint64_t completionToken, id<MTLCommandBuffer> completedBuffer);

#endif
