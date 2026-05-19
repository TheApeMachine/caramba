#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static void metal_utility_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_utility_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_weight_freeze_kernel_name(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "weight_freeze_mask_float32";
    case MetalElementDTypeFloat16: return "weight_freeze_mask_float16";
    case MetalElementDTypeBFloat16: return "weight_freeze_mask_bfloat16";
    default: return NULL;
    }
}

static void metal_utility_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal utility command buffer failed";

        if (error != nil) {
            message = [NSString stringWithFormat:@"%@: %@", message, [error localizedDescription]];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

static int metal_utility_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    void (^encode)(id<MTLComputeCommandEncoder> encoder)
) {
    @autoreleasepool {
        metal_utility_status_clear(status);

        if (threadCount == 0) {
            metal_utility_status_set(status, -6, "empty Metal utility dispatch");
            return -6;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL || context->device == NULL) {
            metal_utility_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, kernelName, status);

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_utility_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_utility_status_set(status, -4, "computeCommandEncoder returned nil");
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
            metal_utility_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_checkpoint_encode_float32(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rank,
    uint32_t count,
    const uint64_t* dims,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_utility_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    uint64_t emptyDim = 0;
    const uint64_t* encodedDims = rank == 0 ? &emptyDim : dims;

    return metal_utility_dispatch(
        contextRef,
        "checkpoint_encode_float32",
        (NSUInteger)(count == 0 ? 1 : count),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&rank length:sizeof(rank) atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
            [encoder setBytes:encodedDims length:sizeof(uint64_t) * (rank == 0 ? 1 : rank) atIndex:4];
        }
    );
}

int metal_dispatch_checkpoint_decode_float32(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t headerBytes,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_utility_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_utility_dispatch(
        contextRef,
        "checkpoint_decode_float32",
        (NSUInteger)(count == 0 ? 1 : count),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&headerBytes length:sizeof(headerBytes) atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
        }
    );
}

int metal_dispatch_tokenizer_pack_int32(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_utility_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_utility_dispatch(
        contextRef,
        "tokenizer_pack_int32",
        (NSUInteger)((count + 3u) / 4u),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&count length:sizeof(count) atIndex:2];
        }
    );
}

int metal_dispatch_weight_freeze_mask(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef maskRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (maskRef == NULL || gradientsRef == NULL || outRef == NULL) {
        metal_utility_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    const char* kernelName = metal_weight_freeze_kernel_name(elementDType);

    if (kernelName == NULL) {
        metal_utility_status_set(status, -6, "unknown Metal weight-freeze kernel");
        return -6;
    }

    return metal_utility_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)((count + 3u) / 4u),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)maskRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)gradientsRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
        }
    );
}
