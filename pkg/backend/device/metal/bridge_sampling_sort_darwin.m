#include "bridge_sampling_private.h"

int metal_sampling_encode_bitonic(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    uint32_t stageSize,
    uint32_t passSize,
    uint32_t paddedCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_sampling_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)indicesRef offset:0 atIndex:1];
    [encoder setBytes:&stageSize length:sizeof(stageSize) atIndex:2];
    [encoder setBytes:&passSize length:sizeof(passSize) atIndex:3];
    [encoder setBytes:&paddedCount length:sizeof(paddedCount) atIndex:4];
    NSUInteger threadWidth = [pipeline threadExecutionWidth];
    if (threadWidth == 0) {
        threadWidth = 1;
    }

    [encoder
        dispatchThreads:MTLSizeMake((NSUInteger)paddedCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_sampling_encode_draw(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t count,
    float target,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_sampling_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)indicesRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    [encoder setBytes:&target length:sizeof(target) atIndex:4];
    [encoder
        dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_sampling_encode_sort(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    uint32_t paddedCount,
    MetalStatus* status
) {
    for (uint32_t stageSize = 2; stageSize <= paddedCount; stageSize <<= 1) {
        for (uint32_t passSize = stageSize >> 1; passSize > 0; passSize >>= 1) {
            int code = metal_sampling_encode_bitonic(
                commandBuffer, pipeline, scoresRef, indicesRef, stageSize, passSize, paddedCount, status
            );

            if (code != 0) {
                return code;
            }
        }

        if (stageSize == paddedCount) {
            break;
        }
    }

    return 0;
}
