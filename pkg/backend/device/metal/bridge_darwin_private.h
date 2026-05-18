#ifndef CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_DARWIN_PRIVATE_H
#define CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_DARWIN_PRIVATE_H

#include "bridge_darwin.h"

#include <Metal/Metal.h>

typedef struct MetalContext {
    void* device;
    void* queue;
    void* library;
    void* pipelineCache;
    void* pipelineLock;
} MetalContext;

id<MTLComputePipelineState> metal_get_pipeline(
    MetalContext* context,
    const char* name,
    MetalStatus* status
);

#endif
