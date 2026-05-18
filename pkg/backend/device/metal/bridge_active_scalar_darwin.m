#include "bridge_active_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static const NSUInteger metalActiveScalarThreadCount = 256;

static void metal_active_scalar_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static int metal_active_scalar_partial_name(
    const char* operationName,
    int elementDType,
    char* partialName,
    MetalStatus* status
) {
    return metal_active_kernel_name(
        partialName, 128, operationName, "partial", elementDType, status
    );
}

static int metal_active_encode_free_energy(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef likelihoodRef,
    MetalBufferRef posteriorRef,
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
    [encoder setBuffer:(__bridge id<MTLBuffer>)posteriorRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)priorRef offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:3];
    [encoder setBytes:&count length:sizeof(count) atIndex:4];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)partialCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalActiveScalarThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

static int metal_active_scalar_dispatch(
    MetalDeviceRef contextRef,
    int elementDType,
    const char* operationName,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status,
    int (^encodePartial)(id<MTLCommandBuffer>, id<MTLComputePipelineState>, MetalStatus*)
) {
    char partialName[128];
    int nameCode = metal_active_scalar_partial_name(
        operationName, elementDType, partialName, status
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

    id<MTLComputePipelineState> partialPipeline = nil;
    int partialPipelineCode = metal_active_pipeline(context, partialName, status, &partialPipeline);

    if (partialPipelineCode != 0) {
        return partialPipelineCode;
    }

    int partialCode = encodePartial(commandBuffer, partialPipeline, status);
    if (partialCode != 0) {
        return partialCode;
    }

    return 0;
}

int metal_dispatch_active_free_energy(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef likelihoodRef,
    MetalBufferRef posteriorRef,
    MetalBufferRef priorRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_active_scalar_status_clear(status);

        if (likelihoodRef == NULL || posteriorRef == NULL || priorRef == NULL ||
            scratchRef == NULL || outRef == NULL) {
            metal_active_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        __block id<MTLCommandBuffer> commandBuffer = nil;
        __block id<MTLComputePipelineState> finalizePipeline = nil;
        int setupCode = metal_active_scalar_dispatch(
            contextRef,
            elementDType,
            "free_energy",
            partialCount,
            completionToken,
            status,
            ^int(id<MTLCommandBuffer> buffer, id<MTLComputePipelineState> pipeline, MetalStatus* localStatus) {
                commandBuffer = buffer;
                return metal_active_encode_free_energy(
                    buffer,
                    pipeline,
                    likelihoodRef,
                    posteriorRef,
                    priorRef,
                    scratchRef,
                    count,
                    partialCount,
                    localStatus
                );
            }
        );

        if (setupCode != 0) {
            return setupCode;
        }

        char finalizeName[128];
        int finalizeNameCode = metal_active_kernel_name(
            finalizeName, sizeof(finalizeName), "active_scalar_finalize", "value", elementDType, status
        );

        if (finalizeNameCode != 0) {
            return finalizeNameCode;
        }

        MetalContext* context = (MetalContext*)contextRef;
        int pipelineCode = metal_active_pipeline(context, finalizeName, status, &finalizePipeline);
        if (pipelineCode != 0) {
            return pipelineCode;
        }

        int finalizeCode = metal_active_encode_finalize(
            commandBuffer, finalizePipeline, scratchRef, outRef, partialCount, status
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
