#include "bridge_hawkes_markov_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

static NSUInteger metal_markov_thread_width(id<MTLComputePipelineState> pipeline) {
    NSUInteger threadWidth = [pipeline threadExecutionWidth];

    if (threadWidth == 0) {
        return 256;
    }

    return threadWidth;
}

static int metal_markov_prepare(
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

int metal_dispatch_markov_blanket_partition(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef adjacencyRef,
    MetalBufferRef internalRef,
    MetalBufferRef outRef,
    uint32_t nodeCount,
    uint32_t internalCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (adjacencyRef == NULL || internalRef == NULL || outRef == NULL) {
            metal_hm_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_markov_prepare(
            contextRef, "markov_blanket_partition", elementDType, status, &commandBuffer, &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
        if (encoderCode != 0) {
            return encoderCode;
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)adjacencyRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)internalRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
        [encoder setBytes:&nodeCount length:sizeof(nodeCount) atIndex:3];
        [encoder setBytes:&internalCount length:sizeof(internalCount) atIndex:4];
        [encoder
            dispatchThreads:MTLSizeMake((NSUInteger)nodeCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(metal_markov_thread_width(pipeline), 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_hm_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_markov_flow(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef mutualInformationRef,
    MetalBufferRef partitionRef,
    MetalBufferRef outRef,
    uint32_t nodeCount,
    int32_t targetLabel,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (mutualInformationRef == NULL || partitionRef == NULL || outRef == NULL) {
            metal_hm_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_markov_prepare(
            contextRef, "markov_flow", elementDType, status, &commandBuffer, &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_hm_encoder(commandBuffer, pipeline, status, &encoder);
        if (encoderCode != 0) {
            return encoderCode;
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)mutualInformationRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)partitionRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
        [encoder setBytes:&nodeCount length:sizeof(nodeCount) atIndex:3];
        [encoder setBytes:&targetLabel length:sizeof(targetLabel) atIndex:4];
        [encoder
            dispatchThreads:MTLSizeMake((NSUInteger)nodeCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(metal_markov_thread_width(pipeline), 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_hm_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
