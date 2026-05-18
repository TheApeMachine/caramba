#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>
#include <string.h>

static const char* metal_physics_extra_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

static void metal_physics_extra_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_physics_extra_status_set(MetalStatus* status, int code, const char* message) {
    if (status == NULL) {
        return;
    }

    status->code = code;
    snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "%s", message == NULL ? "" : message);
}

static void metal_physics_extra_complete(
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

static int metal_physics_extra_kernel_name(
    char* out,
    size_t outBytes,
    const char* operation,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_physics_extra_dtype_suffix(elementDType);
    if (operation == NULL || suffix == NULL) {
        metal_physics_extra_status_set(status, -6, "unknown Metal physics kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operation, suffix);
    if (written <= 0 || (size_t)written >= outBytes) {
        metal_physics_extra_status_set(status, -6, "Metal physics kernel name overflow");
        return -6;
    }

    return 0;
}

static int metal_physics_extra_prefixed_name(
    char* out,
    size_t outBytes,
    int elementDType,
    const char* suffix,
    MetalStatus* status
) {
    const char* prefix = metal_physics_extra_dtype_suffix(elementDType);
    if (prefix == NULL || suffix == NULL) {
        metal_physics_extra_status_set(status, -6, "unknown Metal FFT kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", prefix, suffix);
    if (written <= 0 || (size_t)written >= outBytes) {
        metal_physics_extra_status_set(status, -6, "Metal FFT kernel name overflow");
        return -6;
    }

    return 0;
}

static int metal_physics_extra_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandBuffer>* commandBuffer,
    id<MTLComputePipelineState>* pipeline
) {
    MetalContext* context = (MetalContext*)contextRef;
    if (context == NULL || context->queue == NULL || context->device == NULL) {
        metal_physics_extra_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *pipeline = metal_get_pipeline(context, kernelName, status);
    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
    *commandBuffer = [queue commandBuffer];
    if (*commandBuffer == nil) {
        metal_physics_extra_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

static int metal_physics_extra_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    id<MTLComputeCommandEncoder>* encoder
) {
    *encoder = [commandBuffer computeCommandEncoder];
    if (*encoder == nil) {
        metal_physics_extra_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [*encoder setComputePipelineState:pipeline];
    return 0;
}

static void metal_physics_extra_dispatch(
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

int metal_dispatch_madelung_continuity(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef densityRef,
    MetalBufferRef velocityRef,
    MetalBufferRef spacingRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_physics_extra_status_clear(status);
        if (count == 0) {
            return 0;
        }

        if (densityRef == NULL || velocityRef == NULL || spacingRef == NULL || outRef == NULL) {
            metal_physics_extra_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char kernelName[128];
        int nameCode = metal_physics_extra_kernel_name(
            kernelName, sizeof(kernelName), "madelung_continuity", elementDType, status
        );
        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_physics_extra_prepare(contextRef, kernelName, status, &commandBuffer, &pipeline);
        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_physics_extra_encoder(commandBuffer, pipeline, status, &encoder);
        if (encoderCode != 0) {
            return encoderCode;
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)densityRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)velocityRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)spacingRef offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
        [encoder setBytes:&count length:sizeof(count) atIndex:4];
        metal_physics_extra_dispatch(encoder, pipeline, (NSUInteger)count);
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_physics_extra_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

static bool metal_physics_is_power_of_two(uint32_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

static uint32_t metal_physics_log2(uint32_t value) {
    uint32_t bits = 0;
    while (value > 1) {
        value >>= 1;
        bits++;
    }

    return bits;
}
