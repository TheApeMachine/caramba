#include "bridge_causal_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static void metal_causal_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_causal_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_causal_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

int metal_causal_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_causal_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_causal_status_set(status, -6, "unknown Metal causal kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_causal_status_set(status, -6, "Metal causal kernel name overflow");
        return -6;
    }

    return 0;
}

int metal_causal_prepare(
    MetalDeviceRef contextRef,
    MetalStatus* status,
    MetalContext** context,
    id<MTLCommandBuffer>* commandBuffer
) {
    *context = (MetalContext*)contextRef;

    if (*context == NULL || (*context)->queue == NULL || (*context)->device == NULL) {
        metal_causal_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)(*context)->queue;
    *commandBuffer = [queue commandBuffer];

    if (*commandBuffer == nil) {
        metal_causal_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

int metal_causal_pipeline(
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

int metal_causal_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    id<MTLComputeCommandEncoder>* encoder
) {
    *encoder = [commandBuffer computeCommandEncoder];

    if (*encoder == nil) {
        metal_causal_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [*encoder setComputePipelineState:pipeline];
    return 0;
}

void metal_causal_complete(uint64_t completionToken, id<MTLCommandBuffer> completedBuffer) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal causal command buffer failed";

        if (error != nil) {
            message = [NSString stringWithFormat:@"%@: %@", message, [error localizedDescription]];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

int metal_causal_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    MetalCausalEncodeBlock encode
) {
    @autoreleasepool {
        metal_causal_status_clear(status);

        if (threadCount == 0) {
            return 0;
        }

        MetalContext* context = NULL;
        id<MTLCommandBuffer> commandBuffer = nil;
        int prepareCode = metal_causal_prepare(contextRef, status, &context, &commandBuffer);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputePipelineState> pipeline = nil;
        int pipelineCode = metal_causal_pipeline(context, kernelName, status, &pipeline);

        if (pipelineCode != 0) {
            return pipelineCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_causal_encoder(commandBuffer, pipeline, status, &encoder);

        if (encoderCode != 0) {
            return encoderCode;
        }

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
            metal_causal_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
