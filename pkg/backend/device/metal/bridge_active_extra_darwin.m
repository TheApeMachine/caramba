#include "bridge_active_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static const NSUInteger metalActiveExtraThreadCount = 256;

static void metal_active_extra_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static int metal_active_extra_pipeline_pair(
    MetalContext* context,
    char* firstName,
    char* secondName,
    MetalStatus* status,
    id<MTLComputePipelineState>* firstPipeline,
    id<MTLComputePipelineState>* secondPipeline
) {
    int firstCode = metal_active_pipeline(context, firstName, status, firstPipeline);

    if (firstCode != 0) {
        return firstCode;
    }

    return metal_active_pipeline(context, secondName, status, secondPipeline);
}

static int metal_active_encode_expected_partial(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef predictedObsRef,
    MetalBufferRef preferredObsRef,
    MetalBufferRef predictedStateRef,
    MetalBufferRef scratchRef,
    uint32_t obsCount,
    uint32_t stateCount,
    uint32_t obsPartialCount,
    uint32_t statePartialCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int code = metal_active_encoder(commandBuffer, pipeline, status, &encoder);

    if (code != 0) {
        return code;
    }

    uint32_t totalPartialCount = obsPartialCount + statePartialCount;
    [encoder setBuffer:(__bridge id<MTLBuffer>)predictedObsRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)preferredObsRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)predictedStateRef offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:3];
    [encoder setBytes:&obsCount length:sizeof(obsCount) atIndex:4];
    [encoder setBytes:&stateCount length:sizeof(stateCount) atIndex:5];
    [encoder setBytes:&obsPartialCount length:sizeof(obsPartialCount) atIndex:6];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)totalPartialCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalActiveExtraThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

static int metal_active_encode_belief_partial(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef likelihoodRef,
    MetalBufferRef priorRef,
    MetalBufferRef scratchRef,
    uint32_t count,
    uint32_t partialCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int code = metal_active_encoder(commandBuffer, pipeline, status, &encoder);

    if (code != 0) {
        return code;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)likelihoodRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)priorRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)partialCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalActiveExtraThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

static int metal_active_encode_belief_normalize(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef likelihoodRef,
    MetalBufferRef priorRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int code = metal_active_encoder(commandBuffer, pipeline, status, &encoder);

    if (code != 0) {
        return code;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)likelihoodRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)priorRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
    [encoder setBytes:&count length:sizeof(count) atIndex:4];
    [encoder setBytes:&partialCount length:sizeof(partialCount) atIndex:5];
    [encoder
        dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalActiveExtraThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_dispatch_expected_free_energy(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef predictedObsRef,
    MetalBufferRef preferredObsRef,
    MetalBufferRef predictedStateRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t obsCount,
    uint32_t stateCount,
    uint32_t obsPartialCount,
    uint32_t statePartialCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_active_extra_status_clear(status);

        if (predictedObsRef == NULL || preferredObsRef == NULL || predictedStateRef == NULL ||
            scratchRef == NULL || outRef == NULL) {
            metal_active_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char partialName[128];
        char finalizeName[128];
        int partialNameCode = metal_active_kernel_name(
            partialName, sizeof(partialName), "expected_free_energy", "partial", elementDType, status
        );

        if (partialNameCode != 0) {
            return partialNameCode;
        }

        int finalizeNameCode = metal_active_kernel_name(
            finalizeName, sizeof(finalizeName), "active_scalar_finalize", "value", elementDType, status
        );

        if (finalizeNameCode != 0) {
            return finalizeNameCode;
        }

        MetalContext* context = NULL;
        id<MTLCommandBuffer> commandBuffer = nil;
        int prepareCode = metal_active_prepare(contextRef, status, &context, &commandBuffer);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputePipelineState> partialPipeline = nil;
        id<MTLComputePipelineState> finalizePipeline = nil;
        int pipelineCode = metal_active_extra_pipeline_pair(
            context, partialName, finalizeName, status, &partialPipeline, &finalizePipeline
        );

        if (pipelineCode != 0) {
            return pipelineCode;
        }

        int partialCode = metal_active_encode_expected_partial(
            commandBuffer,
            partialPipeline,
            predictedObsRef,
            preferredObsRef,
            predictedStateRef,
            scratchRef,
            obsCount,
            stateCount,
            obsPartialCount,
            statePartialCount,
            status
        );

        if (partialCode != 0) {
            return partialCode;
        }

        uint32_t totalPartialCount = obsPartialCount + statePartialCount;
        int finalizeCode = metal_active_encode_finalize(
            commandBuffer, finalizePipeline, scratchRef, outRef, totalPartialCount, status
        );

        if (finalizeCode != 0) {
            return finalizeCode;
        }

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_active_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_belief_update(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef likelihoodRef,
    MetalBufferRef priorRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_active_extra_status_clear(status);

        if (likelihoodRef == NULL || priorRef == NULL || scratchRef == NULL || outRef == NULL) {
            metal_active_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char partialName[128];
        char normalizeName[128];
        int partialNameCode = metal_active_kernel_name(
            partialName, sizeof(partialName), "belief_update", "partial", elementDType, status
        );

        if (partialNameCode != 0) {
            return partialNameCode;
        }

        int normalizeNameCode = metal_active_kernel_name(
            normalizeName, sizeof(normalizeName), "belief_update", "normalize", elementDType, status
        );

        if (normalizeNameCode != 0) {
            return normalizeNameCode;
        }

        MetalContext* context = NULL;
        id<MTLCommandBuffer> commandBuffer = nil;
        int prepareCode = metal_active_prepare(contextRef, status, &context, &commandBuffer);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputePipelineState> partialPipeline = nil;
        id<MTLComputePipelineState> normalizePipeline = nil;
        int pipelineCode = metal_active_extra_pipeline_pair(
            context, partialName, normalizeName, status, &partialPipeline, &normalizePipeline
        );

        if (pipelineCode != 0) {
            return pipelineCode;
        }

        int partialCode = metal_active_encode_belief_partial(
            commandBuffer, partialPipeline, likelihoodRef, priorRef, scratchRef, count, partialCount, status
        );

        if (partialCode != 0) {
            return partialCode;
        }

        int normalizeCode = metal_active_encode_belief_normalize(
            commandBuffer,
            normalizePipeline,
            likelihoodRef,
            priorRef,
            scratchRef,
            outRef,
            count,
            partialCount,
            status
        );

        if (normalizeCode != 0) {
            return normalizeCode;
        }

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_active_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
