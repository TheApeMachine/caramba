#include "bridge_causal_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

static const NSUInteger metalCausalScalarThreadCount = 256;

static int metal_causal_two_phase_names(
    char* partialName,
    char* finalizeName,
    size_t nameBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    char baseName[128];
    int nameCode = metal_causal_kernel_name(
        baseName, sizeof(baseName), operationName, elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    int partialWritten = snprintf(partialName, nameBytes, "%s_partial", baseName);
    int finalizeWritten = snprintf(finalizeName, nameBytes, "%s_finalize", baseName);

    if (partialWritten <= 0 || finalizeWritten <= 0 ||
        (size_t)partialWritten >= nameBytes || (size_t)finalizeWritten >= nameBytes) {
        metal_causal_status_set(status, -6, "Metal causal scalar kernel name overflow");
        return -6;
    }

    return 0;
}

static int metal_causal_two_phase_pipelines(
    MetalContext* context,
    const char* partialName,
    const char* finalizeName,
    MetalStatus* status,
    id<MTLComputePipelineState>* partialPipeline,
    id<MTLComputePipelineState>* finalizePipeline
) {
    int partialCode = metal_causal_pipeline(context, partialName, status, partialPipeline);

    if (partialCode != 0) {
        return partialCode;
    }

    return metal_causal_pipeline(context, finalizeName, status, finalizePipeline);
}

static int metal_causal_encode_iv_partial(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef instrumentRef,
    MetalBufferRef treatmentRef,
    MetalBufferRef outcomeRef,
    MetalBufferRef scratchRef,
    uint32_t count,
    uint32_t partialCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int code = metal_causal_encoder(commandBuffer, pipeline, status, &encoder);

    if (code != 0) {
        return code;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)instrumentRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)treatmentRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outcomeRef offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:3];
    [encoder setBytes:&count length:sizeof(count) atIndex:4];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)partialCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalCausalScalarThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

static int metal_causal_encode_iv_finalize(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = nil;
    int code = metal_causal_encoder(commandBuffer, pipeline, status, &encoder);

    if (code != 0) {
        return code;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder setBytes:&partialCount length:sizeof(partialCount) atIndex:3];
    [encoder
        dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalCausalScalarThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_dispatch_iv_estimate(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef instrumentRef,
    MetalBufferRef treatmentRef,
    MetalBufferRef outcomeRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (instrumentRef == NULL || treatmentRef == NULL || outcomeRef == NULL ||
            scratchRef == NULL || outRef == NULL) {
            metal_causal_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char partialName[128];
        char finalizeName[128];
        int nameCode = metal_causal_two_phase_names(
            partialName, finalizeName, sizeof(partialName), "iv_estimate", elementDType, status
        );

        if (nameCode != 0) {
            return nameCode;
        }

        MetalContext* context = NULL;
        id<MTLCommandBuffer> commandBuffer = nil;
        int prepareCode = metal_causal_prepare(contextRef, status, &context, &commandBuffer);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputePipelineState> partialPipeline = nil;
        id<MTLComputePipelineState> finalizePipeline = nil;
        int pipelineCode = metal_causal_two_phase_pipelines(
            context, partialName, finalizeName, status, &partialPipeline, &finalizePipeline
        );

        if (pipelineCode != 0) {
            return pipelineCode;
        }

        int partialCode = metal_causal_encode_iv_partial(
            commandBuffer, partialPipeline, instrumentRef, treatmentRef, outcomeRef,
            scratchRef, count, partialCount, status
        );

        if (partialCode != 0) {
            return partialCode;
        }

        int finalizeCode = metal_causal_encode_iv_finalize(
            commandBuffer, finalizePipeline, scratchRef, outRef, count, partialCount, status
        );

        if (finalizeCode != 0) {
            return finalizeCode;
        }

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_causal_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];
        return 0;
    }
}
