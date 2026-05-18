#include "bridge_causal_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

static const NSUInteger metalCausalDAGThreadCount = 256;

static int metal_causal_dag_name(
    char* out,
    size_t outBytes,
    const char* suffix,
    int elementDType,
    MetalStatus* status
) {
    char baseName[128];
    int nameCode = metal_causal_kernel_name(
        baseName, sizeof(baseName), "dag_markov_factorization", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    int written = snprintf(out, outBytes, "%s_%s", baseName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_causal_status_set(status, -6, "Metal causal DAG kernel name overflow");
        return -6;
    }

    return 0;
}

static int metal_causal_encode_dag_partial(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef conditionalsRef,
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

    [encoder setBuffer:(__bridge id<MTLBuffer>)conditionalsRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)partialCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalCausalDAGThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

static int metal_causal_encode_dag_finalize(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
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
    [encoder setBytes:&partialCount length:sizeof(partialCount) atIndex:2];
    [encoder
        dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalCausalDAGThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_dispatch_dag_markov_factorization(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef conditionalsRef,
    MetalBufferRef parentsRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        (void)parentsRef;
        if (conditionalsRef == NULL || scratchRef == NULL || outRef == NULL) {
            metal_causal_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char partialName[128];
        char finalizeName[128];
        int partialNameCode = metal_causal_dag_name(
            partialName, sizeof(partialName), "partial", elementDType, status
        );

        if (partialNameCode != 0) {
            return partialNameCode;
        }

        int finalizeNameCode = metal_causal_dag_name(
            finalizeName, sizeof(finalizeName), "finalize", elementDType, status
        );

        if (finalizeNameCode != 0) {
            return finalizeNameCode;
        }

        MetalContext* context = NULL;
        id<MTLCommandBuffer> commandBuffer = nil;
        int prepareCode = metal_causal_prepare(contextRef, status, &context, &commandBuffer);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputePipelineState> partialPipeline = nil;
        id<MTLComputePipelineState> finalizePipeline = nil;
        int partialPipelineCode = metal_causal_pipeline(context, partialName, status, &partialPipeline);

        if (partialPipelineCode != 0) {
            return partialPipelineCode;
        }

        int finalizePipelineCode = metal_causal_pipeline(context, finalizeName, status, &finalizePipeline);

        if (finalizePipelineCode != 0) {
            return finalizePipelineCode;
        }

        int partialCode = metal_causal_encode_dag_partial(
            commandBuffer, partialPipeline, conditionalsRef, scratchRef, count, partialCount, status
        );

        if (partialCode != 0) {
            return partialCode;
        }

        int finalizeCode = metal_causal_encode_dag_finalize(
            commandBuffer, finalizePipeline, scratchRef, outRef, partialCount, status
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
