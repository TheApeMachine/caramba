#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static void metal_unary_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_unary_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_unary_float32_kernel_name(int operation) {
    switch (operation) {
    case MetalUnaryFloat32Relu:
        return "relu_float32";
    case MetalUnaryFloat32Abs:
        return "abs_float32";
    case MetalUnaryFloat32Neg:
        return "neg_float32";
    case MetalUnaryFloat32Square:
        return "square_float32";
    case MetalUnaryFloat32Recip:
        return "recip_float32";
    case MetalUnaryFloat32Sqrt:
        return "sqrt_float32";
    case MetalUnaryFloat32Sign:
        return "sign_float32";
    default:
        return NULL;
    }
}

static void metal_unary_complete(uint64_t completionToken, id<MTLCommandBuffer> completedBuffer) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal command buffer failed";

        if (error != nil) {
            message = [NSString
                stringWithFormat:@"%@: %@",
                message,
                [error localizedDescription]
            ];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

int metal_dispatch_unary_float32(
    MetalDeviceRef contextRef,
    int operation,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_unary_status_clear(status);

        if (count == 0) {
            return 0;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL) {
            metal_unary_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        const char* kernelName = metal_unary_float32_kernel_name(operation);

        if (kernelName == NULL) {
            metal_unary_status_set(status, -6, "unknown unary float32 operation");
            return -6;
        }

        if (inputRef == NULL || outRef == NULL) {
            metal_unary_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, kernelName, status);

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLBuffer> input = (__bridge id<MTLBuffer>)inputRef;
        id<MTLBuffer> out = (__bridge id<MTLBuffer>)outRef;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_unary_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_unary_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:out offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        NSUInteger vectorCount = (NSUInteger)((count + 3) / 4);
        MTLSize gridSize = MTLSizeMake(vectorCount, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_unary_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
