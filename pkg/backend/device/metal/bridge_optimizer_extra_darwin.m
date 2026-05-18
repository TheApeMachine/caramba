#include "bridge_optimizer_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

int metal_dispatch_hebbian_step(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef weightsRef,
    MetalBufferRef postRef,
    MetalBufferRef preRef,
    MetalBufferRef outRef,
    uint32_t postCount,
    uint32_t preCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (weightsRef == NULL || postRef == NULL || preRef == NULL || outRef == NULL) {
        metal_optimizer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_optimizer_kernel_name(
        kernelName, sizeof(kernelName), "hebbian_step", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    NSUInteger threadCount = (NSUInteger)postCount * preCount;
    return metal_optimizer_dispatch(
        contextRef, kernelName, threadCount, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)weightsRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)postRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)preRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&postCount length:sizeof(postCount) atIndex:4];
            [encoder setBytes:&preCount length:sizeof(preCount) atIndex:5];
        }
    );
}

int metal_dispatch_lars_step(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef paramsRef,
    MetalBufferRef gradientsRef,
    MetalBufferRef momentumRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t groupCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        if (paramsRef == NULL || gradientsRef == NULL || momentumRef == NULL ||
            scratchRef == NULL || outRef == NULL) {
            metal_optimizer_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char normsName[128];
        int normsNameCode = metal_optimizer_kernel_name(
            normsName, sizeof(normsName), "lars_norms", elementDType, status
        );

        if (normsNameCode != 0) {
            return normsNameCode;
        }

        char stepName[128];
        int stepNameCode = metal_optimizer_kernel_name(
            stepName, sizeof(stepName), "lars_step", elementDType, status
        );

        if (stepNameCode != 0) {
            return stepNameCode;
        }

        id<MTLCommandQueue> queue = nil;
        id<MTLComputePipelineState> normsPipeline = nil;
        int normsPrepare = metal_optimizer_prepare(
            contextRef, normsName, status, &queue, &normsPipeline
        );

        if (normsPrepare != 0) {
            return normsPrepare;
        }

        id<MTLCommandQueue> stepQueue = nil;
        id<MTLComputePipelineState> stepPipeline = nil;
        int stepPrepare = metal_optimizer_prepare(
            contextRef, stepName, status, &stepQueue, &stepPipeline
        );

        if (stepPrepare != 0) {
            return stepPrepare;
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_optimizer_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> normsEncoder = [commandBuffer computeCommandEncoder];

        if (normsEncoder == nil) {
            metal_optimizer_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [normsEncoder setComputePipelineState:normsPipeline];
        [normsEncoder setBuffer:(__bridge id<MTLBuffer>)paramsRef offset:0 atIndex:0];
        [normsEncoder setBuffer:(__bridge id<MTLBuffer>)gradientsRef offset:0 atIndex:1];
        [normsEncoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:2];
        [normsEncoder setBytes:&count length:sizeof(count) atIndex:3];
        [normsEncoder
            dispatchThreadgroups:MTLSizeMake((NSUInteger)groupCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)
        ];
        [normsEncoder endEncoding];

        id<MTLComputeCommandEncoder> stepEncoder = [commandBuffer computeCommandEncoder];

        if (stepEncoder == nil) {
            metal_optimizer_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [stepEncoder setComputePipelineState:stepPipeline];
        [stepEncoder setBuffer:(__bridge id<MTLBuffer>)paramsRef offset:0 atIndex:0];
        [stepEncoder setBuffer:(__bridge id<MTLBuffer>)gradientsRef offset:0 atIndex:1];
        [stepEncoder setBuffer:(__bridge id<MTLBuffer>)momentumRef offset:0 atIndex:2];
        [stepEncoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:3];
        [stepEncoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:4];
        [stepEncoder setBytes:&count length:sizeof(count) atIndex:5];
        [stepEncoder setBytes:&groupCount length:sizeof(groupCount) atIndex:6];

        NSUInteger threadWidth = [stepPipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        [stepEncoder
            dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        [stepEncoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_optimizer_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
