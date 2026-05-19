#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static void metal_dropout_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_dropout_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_dropout_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

static int metal_dropout_kernel_name(
    char* out,
    size_t outBytes,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_dropout_dtype_suffix(elementDType);
    if (suffix == NULL) {
        metal_dropout_status_set(status, -6, "unknown Metal dropout kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "dropout_%s", suffix);
    if (written <= 0 || (size_t)written >= outBytes) {
        metal_dropout_status_set(status, -6, "Metal dropout kernel name overflow");
        return -6;
    }

    return 0;
}

static void metal_dropout_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal dropout command buffer failed";

        if (error != nil) {
            message = [NSString stringWithFormat:@"%@: %@", message, [error localizedDescription]];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

static int metal_dropout_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandBuffer>* commandBuffer,
    id<MTLComputePipelineState>* pipeline
) {
    MetalContext* context = (MetalContext*)contextRef;
    if (context == NULL || context->queue == NULL || context->device == NULL) {
        metal_dropout_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *pipeline = metal_get_pipeline(context, kernelName, status);
    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
    *commandBuffer = [queue commandBuffer];
    if (*commandBuffer == nil) {
        metal_dropout_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

int metal_dispatch_dropout(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    float scale,
    uint32_t threshold,
    uint32_t seed0,
    uint32_t seed1,
    uint32_t seed2,
    uint32_t seed3,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_dropout_status_clear(status);

        if (count == 0) {
            return 0;
        }

        if (inputRef == NULL || outRef == NULL) {
            metal_dropout_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char kernelName[128];
        int nameCode = metal_dropout_kernel_name(kernelName, sizeof(kernelName), elementDType, status);
        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_dropout_prepare(contextRef, kernelName, status, &commandBuffer, &pipeline);
        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            metal_dropout_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        uint32_t seed[4] = {seed0, seed1, seed2, seed3};
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:3];
        [encoder setBytes:&threshold length:sizeof(threshold) atIndex:4];
        [encoder setBytes:&seed length:sizeof(seed) atIndex:5];
        NSUInteger threadWidth = [pipeline threadExecutionWidth];
        if (threadWidth == 0) {
            threadWidth = 1;
        }

        [encoder
            dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_dropout_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
