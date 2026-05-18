#include "bridge_hawkes_markov_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

static const NSUInteger metalHMScalarThreadCount = 256;

static int metal_hm_pipeline(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
) {
    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->device == NULL) {
        metal_hm_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *pipeline = metal_get_pipeline(context, kernelName, status);
    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

static int metal_hm_encode_hawkes_log_partial(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef eventsRef,
    MetalBufferRef totalTimeRef,
    MetalBufferRef baselineRef,
    MetalBufferRef alphaRef,
    MetalBufferRef betaRef,
    MetalBufferRef scratchRef,
    uint32_t eventCount,
    uint32_t partialCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
    if (encoderCode != 0) {
        return encoderCode;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)eventsRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)totalTimeRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)baselineRef offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)alphaRef offset:0 atIndex:3];
    [encoder setBuffer:(__bridge id<MTLBuffer>)betaRef offset:0 atIndex:4];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:5];
    [encoder setBytes:&eventCount length:sizeof(eventCount) atIndex:6];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)partialCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalHMScalarThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

static int metal_hm_encode_hawkes_log_finalize(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scratchRef,
    MetalBufferRef totalTimeRef,
    MetalBufferRef baselineRef,
    MetalBufferRef outRef,
    uint32_t partialCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
    if (encoderCode != 0) {
        return encoderCode;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)totalTimeRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)baselineRef offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
    [encoder setBytes:&partialCount length:sizeof(partialCount) atIndex:4];
    [encoder
        dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalHMScalarThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

static int metal_hm_encode_mi_partial(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef jointRef,
    MetalBufferRef scratchRef,
    uint32_t rows,
    uint32_t cols,
    uint32_t partialCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
    if (encoderCode != 0) {
        return encoderCode;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)jointRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)partialCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalHMScalarThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

static int metal_hm_encode_finalize(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t partialCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
    if (encoderCode != 0) {
        return encoderCode;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
    [encoder setBytes:&partialCount length:sizeof(partialCount) atIndex:2];
    [encoder
        dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalHMScalarThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_dispatch_hawkes_log_likelihood(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef eventsRef,
    MetalBufferRef totalTimeRef,
    MetalBufferRef baselineRef,
    MetalBufferRef alphaRef,
    MetalBufferRef betaRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t eventCount,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (eventsRef == NULL || totalTimeRef == NULL || baselineRef == NULL ||
            alphaRef == NULL || betaRef == NULL || scratchRef == NULL || outRef == NULL) {
            metal_hm_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char partialName[128];
        char finalizeName[128];
        int partialNameCode = metal_hm_phase_kernel_name(
            partialName, sizeof(partialName), "hawkes_log_likelihood", "partial", elementDType, status
        );
        int finalizeNameCode = metal_hm_phase_kernel_name(
            finalizeName, sizeof(finalizeName), "hawkes_log_likelihood", "finalize", elementDType, status
        );

        if (partialNameCode != 0 || finalizeNameCode != 0) {
            return partialNameCode != 0 ? partialNameCode : finalizeNameCode;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> partialPipeline = nil;
        int prepareCode = metal_hm_prepare(contextRef, partialName, status, &commandBuffer, &partialPipeline);
        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputePipelineState> finalizePipeline = nil;
        int finalizePipelineCode = metal_hm_pipeline(contextRef, finalizeName, status, &finalizePipeline);
        if (finalizePipelineCode != 0) {
            return finalizePipelineCode;
        }

        int partialCode = metal_hm_encode_hawkes_log_partial(
            commandBuffer, partialPipeline, eventsRef, totalTimeRef, baselineRef, alphaRef, betaRef,
            scratchRef, eventCount, partialCount, status
        );
        if (partialCode != 0) {
            return partialCode;
        }

        int finalizeCode = metal_hm_encode_hawkes_log_finalize(
            commandBuffer, finalizePipeline, scratchRef, totalTimeRef, baselineRef, outRef, partialCount, status
        );
        if (finalizeCode != 0) {
            return finalizeCode;
        }

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_hm_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_markov_mutual_information(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef jointRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (jointRef == NULL || scratchRef == NULL || outRef == NULL) {
            metal_hm_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char partialName[128];
        char finalizeName[128];
        int partialNameCode = metal_hm_phase_kernel_name(
            partialName, sizeof(partialName), "markov_mutual_information", "partial", elementDType, status
        );
        int finalizeNameCode = metal_hm_kernel_name(
            finalizeName, sizeof(finalizeName), "hawkes_markov_finalize", elementDType, status
        );

        if (partialNameCode != 0 || finalizeNameCode != 0) {
            return partialNameCode != 0 ? partialNameCode : finalizeNameCode;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> partialPipeline = nil;
        int prepareCode = metal_hm_prepare(contextRef, partialName, status, &commandBuffer, &partialPipeline);
        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputePipelineState> finalizePipeline = nil;
        int pipelineCode = metal_hm_pipeline(contextRef, finalizeName, status, &finalizePipeline);
        if (pipelineCode != 0) {
            return pipelineCode;
        }

        int partialCode = metal_hm_encode_mi_partial(
            commandBuffer, partialPipeline, jointRef, scratchRef, rows, cols, partialCount, status
        );
        if (partialCode != 0) {
            return partialCode;
        }

        int finalizeCode = metal_hm_encode_finalize(
            commandBuffer, finalizePipeline, scratchRef, outRef, partialCount, status
        );
        if (finalizeCode != 0) {
            return finalizeCode;
        }

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_hm_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
