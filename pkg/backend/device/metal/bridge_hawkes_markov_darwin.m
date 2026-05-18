#include "bridge_hawkes_markov_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static const NSUInteger metalHMThreadCount = 256;

static void metal_hm_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_hm_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_hm_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

int metal_hm_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_hm_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_hm_status_set(status, -6, "unknown Metal Hawkes/Markov kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);
    if (written <= 0 || (size_t)written >= outBytes) {
        metal_hm_status_set(status, -6, "Metal Hawkes/Markov kernel name overflow");
        return -6;
    }

    return 0;
}

int metal_hm_phase_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* phase,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_hm_dtype_suffix(elementDType);

    if (operationName == NULL || phase == NULL || suffix == NULL) {
        metal_hm_status_set(status, -6, "unknown Metal Hawkes/Markov kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s_%s", operationName, suffix, phase);
    if (written <= 0 || (size_t)written >= outBytes) {
        metal_hm_status_set(status, -6, "Metal Hawkes/Markov kernel name overflow");
        return -6;
    }

    return 0;
}

int metal_hm_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandBuffer>* commandBuffer,
    id<MTLComputePipelineState>* pipeline
) {
    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->queue == NULL || context->device == NULL) {
        metal_hm_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *pipeline = metal_get_pipeline(context, kernelName, status);
    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
    *commandBuffer = [queue commandBuffer];
    if (*commandBuffer == nil) {
        metal_hm_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

int metal_hm_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    id<MTLComputeCommandEncoder>* encoder
) {
    *encoder = [commandBuffer computeCommandEncoder];

    if (*encoder == nil) {
        metal_hm_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [*encoder setComputePipelineState:pipeline];
    return 0;
}

void metal_hm_complete(uint64_t completionToken, id<MTLCommandBuffer> completedBuffer) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal Hawkes/Markov command buffer failed";

        if (error != nil) {
            message = [NSString stringWithFormat:@"%@: %@", message, [error localizedDescription]];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

static NSUInteger metal_hm_thread_width(id<MTLComputePipelineState> pipeline) {
    NSUInteger threadWidth = [pipeline threadExecutionWidth];

    if (threadWidth == 0) {
        return metalHMThreadCount;
    }

    return threadWidth;
}

static int metal_hm_simple_prepare(
    MetalDeviceRef contextRef,
    const char* operationName,
    int elementDType,
    MetalStatus* status,
    id<MTLCommandBuffer>* commandBuffer,
    id<MTLComputePipelineState>* pipeline
) {
    char kernelName[128];
    int nameCode = metal_hm_kernel_name(kernelName, sizeof(kernelName), operationName, elementDType, status);

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_hm_prepare(contextRef, kernelName, status, commandBuffer, pipeline);
}

int metal_dispatch_hawkes_intensity(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef eventsRef,
    MetalBufferRef queryTimesRef,
    MetalBufferRef baselineRef,
    MetalBufferRef alphaRef,
    MetalBufferRef betaRef,
    MetalBufferRef outRef,
    uint32_t eventCount,
    uint32_t queryCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_hm_status_clear(status);

        if (eventsRef == NULL || queryTimesRef == NULL || baselineRef == NULL ||
            alphaRef == NULL || betaRef == NULL || outRef == NULL) {
            metal_hm_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_hm_simple_prepare(
            contextRef, "hawkes_intensity", elementDType, status, &commandBuffer, &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
        if (encoderCode != 0) {
            return encoderCode;
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)eventsRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)queryTimesRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)baselineRef offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)alphaRef offset:0 atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)betaRef offset:0 atIndex:4];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:5];
        [encoder setBytes:&eventCount length:sizeof(eventCount) atIndex:6];
        [encoder
            dispatchThreadgroups:MTLSizeMake((NSUInteger)queryCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(metalHMThreadCount, 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_hm_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_hawkes_kernel_matrix(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef eventsRef,
    MetalBufferRef alphaRef,
    MetalBufferRef betaRef,
    MetalBufferRef outRef,
    uint32_t eventCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_hm_status_clear(status);

        if (eventsRef == NULL || alphaRef == NULL || betaRef == NULL || outRef == NULL) {
            metal_hm_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_hm_simple_prepare(
            contextRef, "hawkes_kernel_matrix", elementDType, status, &commandBuffer, &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
        if (encoderCode != 0) {
            return encoderCode;
        }

        uint32_t total = eventCount * eventCount;
        [encoder setBuffer:(__bridge id<MTLBuffer>)eventsRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)alphaRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)betaRef offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
        [encoder setBytes:&eventCount length:sizeof(eventCount) atIndex:4];
        [encoder
            dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(metal_hm_thread_width(pipeline), 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_hm_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
