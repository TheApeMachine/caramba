#include "bridge_optimizer_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static void metal_optimizer_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_optimizer_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_optimizer_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

static const char* metal_optimizer4_operation_name(int operation) {
    switch (operation) {
    case 0: return "adam_step";
    case 1: return "adamw_step";
    case 2: return "adamax_step";
    default: return NULL;
    }
}

static const char* metal_optimizer3_operation_name(int operation) {
    switch (operation) {
    case 3:
    case 4:
    case 5:
    case 6:
        return "optimizer3";
    default:
        return NULL;
    }
}

static const char* metal_optimizer2_operation_name(int operation) {
    switch (operation) {
    case 7: return "lbfgs_step";
    default: return NULL;
    }
}

int metal_optimizer_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_optimizer_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_optimizer_status_set(status, -6, "unknown Metal optimizer kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_optimizer_status_set(status, -6, "Metal optimizer kernel name overflow");
        return -6;
    }

    return 0;
}

int metal_optimizer_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandQueue>* queue,
    id<MTLComputePipelineState>* pipeline
) {
    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->queue == NULL || context->device == NULL) {
        metal_optimizer_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *queue = (__bridge id<MTLCommandQueue>)context->queue;
    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

void metal_optimizer_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal optimizer command buffer failed";

        if (error != nil) {
            message = [NSString stringWithFormat:@"%@: %@", message, [error localizedDescription]];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

int metal_optimizer_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    MetalOptimizerEncodeBlock encode
) {
    @autoreleasepool {
        metal_optimizer_status_clear(status);

        if (threadCount == 0) {
            return 0;
        }

        id<MTLCommandQueue> queue = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_optimizer_prepare(contextRef, kernelName, status, &queue, &pipeline);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_optimizer_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_optimizer_status_set(status, -4, "computeCommandEncoder returned nil");
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
            metal_optimizer_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_optimizer4(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef firstRef,
    MetalBufferRef secondRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (paramsRef == NULL || gradientsRef == NULL || firstRef == NULL ||
        secondRef == NULL || outRef == NULL) {
        metal_optimizer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_optimizer_kernel_name(
        kernelName,
        sizeof(kernelName),
        metal_optimizer4_operation_name(operation),
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_optimizer_dispatch(
        contextRef, kernelName, (NSUInteger)count, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)paramsRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)gradientsRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)firstRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)secondRef offset:0 atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:4];
            [encoder setBytes:&count length:sizeof(count) atIndex:5];
        }
    );
}

int metal_dispatch_optimizer3(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef stateRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (paramsRef == NULL || gradientsRef == NULL || stateRef == NULL || outRef == NULL) {
        metal_optimizer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_optimizer_kernel_name(
        kernelName,
        sizeof(kernelName),
        metal_optimizer3_operation_name(operation),
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    uint32_t operationCode = (uint32_t)operation;
    return metal_optimizer_dispatch(
        contextRef, kernelName, (NSUInteger)count, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)paramsRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)gradientsRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)stateRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&count length:sizeof(count) atIndex:4];
            [encoder setBytes:&operationCode length:sizeof(operationCode) atIndex:5];
        }
    );
}

int metal_dispatch_optimizer2(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (paramsRef == NULL || gradientsRef == NULL || outRef == NULL) {
        metal_optimizer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_optimizer_kernel_name(
        kernelName,
        sizeof(kernelName),
        metal_optimizer2_operation_name(operation),
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_optimizer_dispatch(
        contextRef, kernelName, (NSUInteger)count, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)paramsRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)gradientsRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
        }
    );
}
