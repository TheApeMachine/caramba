#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static void metal_softmax_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_softmax_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_softmax_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

static int metal_softmax_kernel_name(
    char* out,
    size_t outBytes,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_softmax_dtype_suffix(elementDType);

    if (suffix == NULL) {
        metal_softmax_status_set(status, -6, "unknown Metal softmax kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "softmax_%s", suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_softmax_status_set(status, -6, "Metal softmax kernel name overflow");
        return -6;
    }

    return 0;
}

static void metal_softmax_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal softmax command buffer failed";

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

int metal_dispatch_softmax(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_softmax_status_clear(status);

        if (inputRef == NULL || outRef == NULL) {
            metal_softmax_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL) {
            metal_softmax_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        char kernelName[128];
        int nameCode = metal_softmax_kernel_name(
            kernelName,
            sizeof(kernelName),
            elementDType,
            status
        );

        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, kernelName, status);

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_softmax_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_softmax_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:2];
        [encoder
            dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_softmax_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
