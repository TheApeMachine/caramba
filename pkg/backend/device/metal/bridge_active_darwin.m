#include "bridge_active_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static const NSUInteger metalActiveThreadCount = 256;

static void metal_active_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_active_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_active_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

int metal_active_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* phase,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_active_dtype_suffix(elementDType);

    if (operationName == NULL || phase == NULL || suffix == NULL) {
        metal_active_status_set(status, -6, "unknown Metal active inference kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s_%s", operationName, suffix, phase);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_active_status_set(status, -6, "Metal active inference kernel name overflow");
        return -6;
    }

    return 0;
}

static int metal_active_single_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_active_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_active_status_set(status, -6, "unknown Metal active inference kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_active_status_set(status, -6, "Metal active inference kernel name overflow");
        return -6;
    }

    return 0;
}

int metal_active_prepare(
    MetalDeviceRef contextRef,
    MetalStatus* status,
    MetalContext** context,
    id<MTLCommandBuffer>* commandBuffer
) {
    *context = (MetalContext*)contextRef;

    if (*context == NULL || (*context)->queue == NULL) {
        metal_active_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)(*context)->queue;
    *commandBuffer = [queue commandBuffer];

    if (*commandBuffer == nil) {
        metal_active_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

int metal_active_pipeline(
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

int metal_active_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    id<MTLComputeCommandEncoder>* encoder
) {
    *encoder = [commandBuffer computeCommandEncoder];

    if (*encoder == nil) {
        metal_active_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [*encoder setComputePipelineState:pipeline];
    return 0;
}

void metal_active_complete(uint64_t completionToken, id<MTLCommandBuffer> completedBuffer) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal active inference command buffer failed";

        if (error != nil) {
            message = [NSString stringWithFormat:@"%@: %@", message, [error localizedDescription]];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

int metal_active_encode_finalize(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t partialCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int code = metal_active_encoder(commandBuffer, pipeline, status, &encoder);

    if (code != 0) {
        return code;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
    [encoder setBytes:&partialCount length:sizeof(partialCount) atIndex:2];
    [encoder
        dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalActiveThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

static int metal_active_encode_precision(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef errorsRef,
    MetalBufferRef precisionRef,
    MetalBufferRef outRef,
    uint32_t count,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int code = metal_active_encoder(commandBuffer, pipeline, status, &encoder);

    if (code != 0) {
        return code;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)errorsRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)precisionRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    NSUInteger threadWidth = [pipeline threadExecutionWidth];
    if (threadWidth == 0) {
        threadWidth = metalActiveThreadCount;
    }

    [encoder
        dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_dispatch_precision_weight(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef errorsRef,
    MetalBufferRef precisionRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_active_status_clear(status);

        if (errorsRef == NULL || precisionRef == NULL || outRef == NULL) {
            metal_active_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char kernelName[128];
        int nameCode = metal_active_single_kernel_name(
            kernelName, sizeof(kernelName), "precision_weight", elementDType, status
        );

        if (nameCode != 0) {
            return nameCode;
        }

        MetalContext* context = NULL;
        id<MTLCommandBuffer> commandBuffer = nil;
        int prepareCode = metal_active_prepare(contextRef, status, &context, &commandBuffer);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputePipelineState> pipeline = nil;
        int pipelineCode = metal_active_pipeline(context, kernelName, status, &pipeline);

        if (pipelineCode != 0) {
            return pipelineCode;
        }

        int encodeCode = metal_active_encode_precision(
            commandBuffer, pipeline, errorsRef, precisionRef, outRef, count, status
        );

        if (encodeCode != 0) {
            return encodeCode;
        }

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_active_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
