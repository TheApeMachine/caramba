#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static const NSUInteger metalReductionThreadCountObjC = 256;

static void metal_reduction_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_reduction_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_reduction_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

static int metal_reduction_kernel_name(
    char* out,
    size_t outBytes,
    const char* phase,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_reduction_dtype_suffix(elementDType);

    if (phase == NULL || suffix == NULL) {
        metal_reduction_status_set(status, -6, "unknown Metal reduction kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "reduction_%s_%s", phase, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_reduction_status_set(status, -6, "Metal reduction kernel name overflow");
        return -6;
    }

    return 0;
}

static void metal_reduction_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal reduction command buffer failed";

        if (error != nil) {
            message = [NSString stringWithFormat:@"%@: %@", message, [error localizedDescription]];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

static int metal_reduction_prepare(
    MetalDeviceRef contextRef,
    MetalStatus* status,
    MetalContext** context,
    id<MTLCommandBuffer>* commandBuffer
) {
    *context = (MetalContext*)contextRef;

    if (*context == NULL || (*context)->queue == NULL || (*context)->device == NULL) {
        metal_reduction_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)(*context)->queue;
    *commandBuffer = [queue commandBuffer];

    if (*commandBuffer == nil) {
        metal_reduction_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

static int metal_reduction_pipeline(
    MetalContext* context,
    const char* kernelName,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
) {
    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

static int metal_reduction_encode_partial(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef inputRef,
    MetalBufferRef scratchARef,
    MetalBufferRef scratchBRef,
    uint32_t count,
    uint32_t partialCount,
    uint32_t operation,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_reduction_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchARef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchBRef offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    [encoder setBytes:&operation length:sizeof(operation) atIndex:4];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)partialCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalReductionThreadCountObjC, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

static int metal_reduction_encode_finalize(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scratchARef,
    MetalBufferRef scratchBRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint32_t operation,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_reduction_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchARef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchBRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
    [encoder setBytes:&partialCount length:sizeof(partialCount) atIndex:3];
    [encoder setBytes:&count length:sizeof(count) atIndex:4];
    [encoder setBytes:&operation length:sizeof(operation) atIndex:5];
    [encoder
        dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalReductionThreadCountObjC, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_dispatch_reduction(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scratchARef,
    MetalBufferRef scratchBRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_reduction_status_clear(status);

        if (count == 0 || partialCount == 0) {
            return 0;
        }

        if (inputRef == NULL || scratchARef == NULL || scratchBRef == NULL || outRef == NULL) {
            metal_reduction_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char partialName[128];
        char finalizeName[128];
        int partialNameCode = metal_reduction_kernel_name(
            partialName, sizeof(partialName), "partial", elementDType, status
        );

        if (partialNameCode != 0) {
            return partialNameCode;
        }

        int finalizeNameCode = metal_reduction_kernel_name(
            finalizeName, sizeof(finalizeName), "finalize", elementDType, status
        );

        if (finalizeNameCode != 0) {
            return finalizeNameCode;
        }

        MetalContext* context = NULL;
        id<MTLCommandBuffer> commandBuffer = nil;
        int prepareCode = metal_reduction_prepare(contextRef, status, &context, &commandBuffer);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputePipelineState> partialPipeline = nil;
        int partialPipelineCode = metal_reduction_pipeline(
            context, partialName, status, &partialPipeline
        );

        if (partialPipelineCode != 0) {
            return partialPipelineCode;
        }

        id<MTLComputePipelineState> finalizePipeline = nil;
        int finalizePipelineCode = metal_reduction_pipeline(
            context, finalizeName, status, &finalizePipeline
        );

        if (finalizePipelineCode != 0) {
            return finalizePipelineCode;
        }

        uint32_t operationCode = (uint32_t)operation;
        int partialCode = metal_reduction_encode_partial(
            commandBuffer,
            partialPipeline,
            inputRef,
            scratchARef,
            scratchBRef,
            count,
            partialCount,
            operationCode,
            status
        );

        if (partialCode != 0) {
            return partialCode;
        }

        int finalizeCode = metal_reduction_encode_finalize(
            commandBuffer,
            finalizePipeline,
            scratchARef,
            scratchBRef,
            outRef,
            count,
            partialCount,
            operationCode,
            status
        );

        if (finalizeCode != 0) {
            return finalizeCode;
        }

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_reduction_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
