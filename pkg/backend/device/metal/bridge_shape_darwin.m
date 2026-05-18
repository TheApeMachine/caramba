#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static void metal_shape_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_shape_status_set(MetalStatus* status, int code, const char* message) {
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

static void metal_shape_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal shape command buffer failed";

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

static int metal_shape_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    void (^encode)(id<MTLComputeCommandEncoder> encoder)
) {
    @autoreleasepool {
        metal_shape_status_clear(status);

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL) {
            metal_shape_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        if (threadCount == 0) {
            metal_shape_status_set(status, -6, "empty Metal shape dispatch");
            return -6;
        }

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, kernelName, status);

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_shape_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_shape_status_set(status, -4, "computeCommandEncoder returned nil");
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
            metal_shape_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_copy_bytes(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t byteCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_shape_dispatch(
        contextRef,
        "copy_bytes",
        (NSUInteger)((byteCount + 15) / 16),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&byteCount length:sizeof(byteCount) atIndex:2];
        }
    );
}

int metal_dispatch_concat_bytes(
    MetalDeviceRef contextRef,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t leftBytes,
    uint32_t rightBytes,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (leftRef == NULL || rightRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    uint32_t totalBytes = leftBytes + rightBytes;

    return metal_shape_dispatch(
        contextRef,
        "concat_bytes",
        (NSUInteger)((totalBytes + 15) / 16),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)leftRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)rightRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&leftBytes length:sizeof(leftBytes) atIndex:3];
            [encoder setBytes:&totalBytes length:sizeof(totalBytes) atIndex:4];
        }
    );
}

int metal_dispatch_split2_bytes(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    uint32_t leftBytes,
    uint32_t rightBytes,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || leftRef == NULL || rightRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    uint32_t totalBytes = leftBytes + rightBytes;

    return metal_shape_dispatch(
        contextRef,
        "split2_bytes",
        (NSUInteger)((totalBytes + 15) / 16),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)leftRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)rightRef offset:0 atIndex:2];
            [encoder setBytes:&leftBytes length:sizeof(leftBytes) atIndex:3];
            [encoder setBytes:&totalBytes length:sizeof(totalBytes) atIndex:4];
        }
    );
}

int metal_dispatch_last_token_bytes(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t seq,
    uint32_t hiddenBytes,
    uint32_t outBytes,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_shape_dispatch(
        contextRef,
        "last_token_bytes",
        (NSUInteger)((outBytes + 15) / 16),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&seq length:sizeof(seq) atIndex:2];
            [encoder setBytes:&hiddenBytes length:sizeof(hiddenBytes) atIndex:3];
            [encoder setBytes:&outBytes length:sizeof(outBytes) atIndex:4];
        }
    );
}

int metal_dispatch_transpose2d_bytes(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint32_t elementBytes,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_shape_dispatch(
        contextRef,
        "transpose2d_bytes",
        (NSUInteger)(rows * cols),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
            [encoder setBytes:&elementBytes length:sizeof(elementBytes) atIndex:4];
        }
    );
}

int metal_dispatch_upsample_nearest2d_bytes(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t channels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint32_t elementBytes,
    uint32_t outElements,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_shape_dispatch(
        contextRef,
        "upsample_nearest2d_bytes",
        (NSUInteger)outElements,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&channels length:sizeof(channels) atIndex:2];
            [encoder setBytes:&inHeight length:sizeof(inHeight) atIndex:3];
            [encoder setBytes:&inWidth length:sizeof(inWidth) atIndex:4];
            [encoder setBytes:&outHeight length:sizeof(outHeight) atIndex:5];
            [encoder setBytes:&outWidth length:sizeof(outWidth) atIndex:6];
            [encoder setBytes:&elementBytes length:sizeof(elementBytes) atIndex:7];
            [encoder setBytes:&outElements length:sizeof(outElements) atIndex:8];
        }
    );
}
