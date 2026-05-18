#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>
#include <string.h>

static void metal_physics_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_physics_status_set(MetalStatus* status, int code, const char* message) {
    if (status == NULL) {
        return;
    }

    status->code = code;

    if (message == NULL) {
        status->message[0] = '\0';
        return;
    }

    snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "%s", message);
}

static const char* metal_physics_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

static int metal_physics_kernel_name(
    char* out,
    size_t outBytes,
    const char* operation,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_physics_dtype_suffix(elementDType);
    if (operation == NULL || suffix == NULL) {
        metal_physics_status_set(status, -6, "unknown Metal physics kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operation, suffix);
    if (written <= 0 || (size_t)written >= outBytes) {
        metal_physics_status_set(status, -6, "Metal physics kernel name overflow");
        return -6;
    }

    return 0;
}

static int metal_physics_prefixed_name(
    char* out,
    size_t outBytes,
    int elementDType,
    const char* suffix,
    MetalStatus* status
) {
    const char* prefix = metal_physics_dtype_suffix(elementDType);
    if (prefix == NULL || suffix == NULL) {
        metal_physics_status_set(status, -6, "unknown Metal physics FFT kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", prefix, suffix);
    if (written <= 0 || (size_t)written >= outBytes) {
        metal_physics_status_set(status, -6, "Metal physics FFT kernel name overflow");
        return -6;
    }

    return 0;
}

static void metal_physics_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal physics command buffer failed";
        if (error != nil) {
            message = [NSString stringWithFormat:@"%@: %@", message, [error localizedDescription]];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

static int metal_physics_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandBuffer>* commandBuffer,
    id<MTLComputePipelineState>* pipeline
) {
    MetalContext* context = (MetalContext*)contextRef;
    if (context == NULL || context->queue == NULL || context->device == NULL) {
        metal_physics_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *pipeline = metal_get_pipeline(context, kernelName, status);
    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
    *commandBuffer = [queue commandBuffer];
    if (*commandBuffer == nil) {
        metal_physics_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

static int metal_physics_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    id<MTLComputeCommandEncoder>* encoder
) {
    *encoder = [commandBuffer computeCommandEncoder];
    if (*encoder == nil) {
        metal_physics_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [*encoder setComputePipelineState:pipeline];
    return 0;
}

static void metal_physics_dispatch_threads(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    NSUInteger count
) {
    NSUInteger threadWidth = [pipeline threadExecutionWidth];
    if (threadWidth == 0) {
        threadWidth = 1;
    }

    [encoder
        dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
    ];
}

static int metal_dispatch_physics_vector(
    MetalDeviceRef contextRef,
    int elementDType,
    const char* operation,
    MetalBufferRef inputRef,
    MetalBufferRef spacingRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_physics_status_clear(status);
        if (count == 0) {
            return 0;
        }

        if (inputRef == NULL || spacingRef == NULL || outRef == NULL) {
            metal_physics_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char kernelName[128];
        int nameCode = metal_physics_kernel_name(kernelName, sizeof(kernelName), operation, elementDType, status);
        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_physics_prepare(contextRef, kernelName, status, &commandBuffer, &pipeline);
        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_physics_encoder(commandBuffer, pipeline, status, &encoder);
        if (encoderCode != 0) {
            return encoderCode;
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)spacingRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
        [encoder setBytes:&count length:sizeof(count) atIndex:3];
        metal_physics_dispatch_threads(encoder, pipeline, (NSUInteger)count);
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_physics_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_laplacian(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef spacingRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t rank,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_physics_status_clear(status);
        if (count == 0) {
            return 0;
        }

        if (inputRef == NULL || spacingRef == NULL || outRef == NULL) {
            metal_physics_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char kernelName[128];
        int nameCode = metal_physics_kernel_name(kernelName, sizeof(kernelName), "laplacian", elementDType, status);
        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_physics_prepare(contextRef, kernelName, status, &commandBuffer, &pipeline);
        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_physics_encoder(commandBuffer, pipeline, status, &encoder);
        if (encoderCode != 0) {
            return encoderCode;
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)spacingRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
        [encoder setBytes:&count length:sizeof(count) atIndex:3];
        [encoder setBytes:&rank length:sizeof(rank) atIndex:4];
        [encoder setBytes:&dim0 length:sizeof(dim0) atIndex:5];
        [encoder setBytes:&dim1 length:sizeof(dim1) atIndex:6];
        [encoder setBytes:&dim2 length:sizeof(dim2) atIndex:7];
        metal_physics_dispatch_threads(encoder, pipeline, (NSUInteger)count);
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_physics_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

#define PHYSICS_VECTOR_DISPATCH(name, operation) \
int name( \
    MetalDeviceRef contextRef, int elementDType, MetalBufferRef inputRef, \
    MetalBufferRef spacingRef, MetalBufferRef outRef, uint32_t count, \
    uint64_t completionToken, MetalStatus* status \
) { \
    return metal_dispatch_physics_vector( \
        contextRef, elementDType, operation, inputRef, spacingRef, outRef, \
        count, completionToken, status \
    ); \
}

PHYSICS_VECTOR_DISPATCH(metal_dispatch_laplacian4, "laplacian4")
PHYSICS_VECTOR_DISPATCH(metal_dispatch_grad1d, "grad1d")
PHYSICS_VECTOR_DISPATCH(metal_dispatch_divergence1d, "divergence1d")
PHYSICS_VECTOR_DISPATCH(metal_dispatch_quantum_potential, "quantum_potential")
PHYSICS_VECTOR_DISPATCH(metal_dispatch_bohmian_velocity, "bohmian_velocity")
