#include "bridge_vision_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>
#include <string.h>

static void metal_vision_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_vision_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_vision_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

int metal_vision_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_vision_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_vision_status_set(status, -6, "unknown Metal vision kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_vision_status_set(status, -6, "Metal vision kernel name overflow");
        return -6;
    }

    return 0;
}

static int metal_vision_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandQueue>* queue,
    id<MTLComputePipelineState>* pipeline
) {
    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->queue == NULL || context->device == NULL) {
        metal_vision_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *queue = (__bridge id<MTLCommandQueue>)context->queue;
    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

int metal_vision_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    MetalVisionEncodeBlock encode
) {
    @autoreleasepool {
        metal_vision_status_clear(status);

        if (threadCount == 0) {
            return 0;
        }

        id<MTLCommandQueue> queue = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_vision_prepare(contextRef, kernelName, status, &queue, &pipeline);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_vision_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_vision_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [encoder setComputePipelineState:pipeline];
        encode(encoder);

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        [encoder
            dispatchThreads:MTLSizeMake(threadCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            @autoreleasepool {
                if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
                    metalCommandCompleted(completionToken, 0, "");
                    return;
                }

                NSError* error = [completedBuffer error];
                NSString* message = @"Metal vision command buffer failed";

                if (error != nil) {
                    message = [NSString stringWithFormat:@"%@: %@", message, [error localizedDescription]];
                }

                metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
            }
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_conv2d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef weightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t inChannels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outChannels,
    uint32_t kernelHeight,
    uint32_t kernelWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || weightRef == NULL || biasRef == NULL || outRef == NULL) {
        metal_vision_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_vision_kernel_name(
        kernelName, sizeof(kernelName), "conv2d", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    NSUInteger threadCount = (NSUInteger)batch * outChannels * outHeight * outWidth;
    return metal_vision_dispatch(
        contextRef, kernelName, threadCount, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)weightRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:4];
            [encoder setBytes:&inChannels length:sizeof(inChannels) atIndex:5];
            [encoder setBytes:&inHeight length:sizeof(inHeight) atIndex:6];
            [encoder setBytes:&inWidth length:sizeof(inWidth) atIndex:7];
            [encoder setBytes:&outChannels length:sizeof(outChannels) atIndex:8];
            [encoder setBytes:&kernelHeight length:sizeof(kernelHeight) atIndex:9];
            [encoder setBytes:&kernelWidth length:sizeof(kernelWidth) atIndex:10];
            [encoder setBytes:&outHeight length:sizeof(outHeight) atIndex:11];
            [encoder setBytes:&outWidth length:sizeof(outWidth) atIndex:12];
        }
    );
}

int metal_dispatch_pool2d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    bool useMax,
    bool adaptive,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_vision_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    const char* operationName = adaptive ?
        (useMax ? "adaptive_max_pool2d" : "adaptive_avg_pool2d") :
        (useMax ? "max_pool2d" : "avg_pool2d");
    char kernelName[128];
    int nameCode = metal_vision_kernel_name(
        kernelName, sizeof(kernelName), operationName, elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    NSUInteger threadCount = (NSUInteger)batch * channels * outHeight * outWidth;

    return metal_vision_dispatch(
        contextRef, kernelName, threadCount, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:2];
            [encoder setBytes:&channels length:sizeof(channels) atIndex:3];
            [encoder setBytes:&inHeight length:sizeof(inHeight) atIndex:4];
            [encoder setBytes:&inWidth length:sizeof(inWidth) atIndex:5];
            [encoder setBytes:&outHeight length:sizeof(outHeight) atIndex:6];
            [encoder setBytes:&outWidth length:sizeof(outWidth) atIndex:7];
            [encoder setBytes:&useMax length:sizeof(useMax) atIndex:8];
        }
    );
}
