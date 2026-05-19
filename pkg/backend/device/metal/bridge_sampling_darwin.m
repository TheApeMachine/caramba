#include "bridge_sampling_private.h"

static int metal_sampling_dispatch_greedy(
    MetalContext* context,
    id<MTLCommandBuffer> commandBuffer,
    int elementDType,
    MetalBufferRef logitsRef,
    MetalBufferRef outRef,
    uint32_t count,
    MetalStatus* status
) {
    char kernelName[128];
    int nameCode = metal_sampling_kernel_name(
        kernelName, sizeof(kernelName), "greedy_sample", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    id<MTLComputePipelineState> pipeline = nil;
    int pipelineCode = metal_sampling_pipeline(context, kernelName, status, &pipeline);

    if (pipelineCode != 0) {
        return pipelineCode;
    }

    return metal_sampling_encode_greedy(commandBuffer, pipeline, logitsRef, outRef, count, status);
}

static int metal_sampling_dispatch_draw(
    MetalContext* context,
    id<MTLCommandBuffer> commandBuffer,
    int elementDType,
    MetalBufferRef logitsRef,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t paddedCount,
    float target,
    MetalStatus* status
) {
    char initName[128];
    int nameCode = metal_sampling_kernel_name(
        initName, sizeof(initName), "sampling_init", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    id<MTLComputePipelineState> initPipeline = nil;
    int initCode = metal_sampling_pipeline(context, initName, status, &initPipeline);

    if (initCode != 0) {
        return initCode;
    }

    id<MTLComputePipelineState> bitonicPipeline = nil;
    int bitonicCode = metal_sampling_pipeline(context, "sampling_bitonic_step", status, &bitonicPipeline);

    if (bitonicCode != 0) {
        return bitonicCode;
    }

    id<MTLComputePipelineState> drawPipeline = nil;
    int drawCode = metal_sampling_pipeline(context, "sampling_draw_sorted", status, &drawPipeline);

    if (drawCode != 0) {
        return drawCode;
    }

    int encodeInitCode = metal_sampling_encode_init(
        commandBuffer, initPipeline, logitsRef, scoresRef, indicesRef, count, paddedCount, status
    );

    if (encodeInitCode != 0) {
        return encodeInitCode;
    }

    int sortCode = metal_sampling_encode_sort(
        commandBuffer, bitonicPipeline, scoresRef, indicesRef, paddedCount, status
    );

    if (sortCode != 0) {
        return sortCode;
    }

    return metal_sampling_encode_draw(
        commandBuffer, drawPipeline, scoresRef, indicesRef, outRef, count, target, status
    );
}

int metal_dispatch_sampling(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef logitsRef,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t paddedCount,
    float target,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_sampling_status_clear(status);

        if (count == 0) {
            return 0;
        }

        if (logitsRef == NULL || outRef == NULL) {
            metal_sampling_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        if (operation != 0 && (scoresRef == NULL || indicesRef == NULL)) {
            metal_sampling_status_set(status, -2, "nil Metal sampling scratch buffer");
            return -2;
        }

        MetalContext* context = NULL;
        id<MTLCommandBuffer> commandBuffer = nil;
        int prepareCode = metal_sampling_prepare(contextRef, status, &context, &commandBuffer);

        if (prepareCode != 0) {
            return prepareCode;
        }

        int dispatchCode = 0;
        if (operation == 0) {
            dispatchCode = metal_sampling_dispatch_greedy(
                context, commandBuffer, elementDType, logitsRef, outRef, count, status
            );
        }

        if (operation != 0) {
            dispatchCode = metal_sampling_dispatch_draw(
                context,
                commandBuffer,
                elementDType,
                logitsRef,
                scoresRef,
                indicesRef,
                outRef,
                count,
                paddedCount,
                target,
                status
            );
        }

        if (dispatchCode != 0) {
            return dispatchCode;
        }

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_sampling_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
