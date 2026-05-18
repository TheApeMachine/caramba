#include "bridge_transformer_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

void metal_attention_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_attention_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal attention command buffer failed";

        if (error != nil) {
            message = [NSString
                stringWithFormat:@"%@: %@",
                message,
                [error localizedDescription]
            ];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

int metal_attention_pipeline(
    MetalContext* context,
    const char* operationName,
    int elementDType,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
) {
    char kernelName[128];
    int nameCode = metal_transformer_kernel_name(
        kernelName, sizeof(kernelName), operationName, elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

static int metal_attention_softmax_pipeline(
    MetalContext* context,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
) {
    *pipeline = metal_get_pipeline(context, "attention_softmax", status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

id<MTLCommandBuffer> metal_attention_command_buffer(
    MetalContext* context,
    MetalStatus* status
) {
    if (context == NULL || context->queue == NULL || context->device == NULL) {
        metal_transformer_status_set(status, -1, "invalid Metal attention context");
        return nil;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

    if (commandBuffer == nil) {
        metal_transformer_status_set(status, -3, "commandBuffer returned nil");
        return nil;
    }

    return commandBuffer;
}

id<MTLComputeCommandEncoder> metal_attention_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_transformer_status_set(status, -4, "computeCommandEncoder returned nil");
        return nil;
    }

    [encoder setComputePipelineState:pipeline];
    return encoder;
}

static int metal_attention_prepare(
    MetalContext* context,
    int elementDType,
    MetalStatus* status,
    id<MTLComputePipelineState>* scoresPipeline,
    id<MTLComputePipelineState>* softmaxPipeline,
    id<MTLComputePipelineState>* weightedPipeline
) {
    int scoresCode = metal_attention_pipeline(
        context, "attention_scores", elementDType, status, scoresPipeline
    );

    if (scoresCode != 0) {
        return scoresCode;
    }

    int softmaxCode = metal_attention_softmax_pipeline(context, status, softmaxPipeline);

    if (softmaxCode != 0) {
        return softmaxCode;
    }

    return metal_attention_pipeline(
        context, "attention_weighted", elementDType, status, weightedPipeline
    );
}

static int metal_attention_encode_scores(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    MetalBufferRef queryRef,
    MetalBufferRef keyRef,
    MetalBufferRef scoresRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t depth
) {
    id<MTLComputeCommandEncoder> encoder =
        metal_attention_encoder(commandBuffer, pipeline, status);

    if (encoder == nil) {
        return status != NULL && status->code != 0 ? status->code : -4;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)queryRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)keyRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:2];
    [encoder setBytes:&seqQ length:sizeof(seqQ) atIndex:3];
    [encoder setBytes:&seqK length:sizeof(seqK) atIndex:4];
    [encoder setBytes:&depth length:sizeof(depth) atIndex:5];
    [encoder
        dispatchThreadgroups:MTLSizeMake((seqK + 15) / 16, (seqQ + 15) / 16, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)
    ];
    [encoder endEncoding];
    return 0;
}

static int metal_attention_encode_softmax(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    MetalBufferRef scoresRef,
    uint32_t seqQ,
    uint32_t seqK
) {
    id<MTLComputeCommandEncoder> encoder =
        metal_attention_encoder(commandBuffer, pipeline, status);

    if (encoder == nil) {
        return status != NULL && status->code != 0 ? status->code : -4;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:0];
    [encoder setBytes:&seqK length:sizeof(seqK) atIndex:1];
    [encoder
        dispatchThreadgroups:MTLSizeMake(seqQ, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)
    ];
    [encoder endEncoding];
    return 0;
}

static int metal_attention_encode_weighted(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    MetalBufferRef valueRef,
    MetalBufferRef scoresRef,
    MetalBufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t valueDim
) {
    id<MTLComputeCommandEncoder> encoder =
        metal_attention_encoder(commandBuffer, pipeline, status);

    if (encoder == nil) {
        return status != NULL && status->code != 0 ? status->code : -4;
    }

    [encoder setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)valueRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
    [encoder setBytes:&seqQ length:sizeof(seqQ) atIndex:3];
    [encoder setBytes:&seqK length:sizeof(seqK) atIndex:4];
    [encoder setBytes:&valueDim length:sizeof(valueDim) atIndex:5];
    [encoder
        dispatchThreadgroups:MTLSizeMake((valueDim + 15) / 16, (seqQ + 15) / 16, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)
    ];
    [encoder endEncoding];
    return 0;
}

int metal_dispatch_attention(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef queryRef,
    MetalBufferRef keyRef,
    MetalBufferRef valueRef,
    MetalBufferRef scoresRef,
    MetalBufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t depth,
    uint32_t valueDim,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_attention_status_clear(status);

        if (queryRef == NULL || keyRef == NULL || valueRef == NULL ||
            scoresRef == NULL || outRef == NULL) {
            metal_transformer_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        MetalContext* context = (MetalContext*)contextRef;
        id<MTLComputePipelineState> scoresPipeline = nil;
        id<MTLComputePipelineState> softmaxPipeline = nil;
        id<MTLComputePipelineState> weightedPipeline = nil;
        int prepareCode = metal_attention_prepare(
            context,
            elementDType,
            status,
            &scoresPipeline,
            &softmaxPipeline,
            &weightedPipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLCommandBuffer> commandBuffer = metal_attention_command_buffer(context, status);

        if (commandBuffer == nil) {
            return status != NULL && status->code != 0 ? status->code : -3;
        }

        int scoresCode = metal_attention_encode_scores(
            commandBuffer, scoresPipeline, status,
            queryRef, keyRef, scoresRef,
            seqQ, seqK, depth
        );

        if (scoresCode != 0) {
            return scoresCode;
        }

        int softmaxCode = metal_attention_encode_softmax(
            commandBuffer, softmaxPipeline, status, scoresRef, seqQ, seqK
        );

        if (softmaxCode != 0) {
            return softmaxCode;
        }

        int weightedCode = metal_attention_encode_weighted(
            commandBuffer, weightedPipeline, status,
            valueRef, scoresRef, outRef,
            seqQ, seqK, valueDim
        );

        if (weightedCode != 0) {
            return weightedCode;
        }

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_attention_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];
        return 0;
    }
}

int metal_dispatch_flash_attention(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef queryRef,
    MetalBufferRef keyRef,
    MetalBufferRef valueRef,
    MetalBufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t depth,
    uint32_t valueDim,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_attention_status_clear(status);

        if (queryRef == NULL || keyRef == NULL || valueRef == NULL || outRef == NULL) {
            metal_transformer_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        MetalContext* context = (MetalContext*)contextRef;
        id<MTLComputePipelineState> pipeline = nil;
        int pipelineCode = metal_attention_pipeline(
            context, "flash_attention", elementDType, status, &pipeline
        );

        if (pipelineCode != 0) {
            return pipelineCode;
        }

        id<MTLCommandBuffer> commandBuffer = metal_attention_command_buffer(context, status);

        if (commandBuffer == nil) {
            return status != NULL && status->code != 0 ? status->code : -3;
        }

        id<MTLComputeCommandEncoder> encoder =
            metal_attention_encoder(commandBuffer, pipeline, status);

        if (encoder == nil) {
            return status != NULL && status->code != 0 ? status->code : -4;
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)queryRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)keyRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)valueRef offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
        [encoder setBytes:&seqQ length:sizeof(seqQ) atIndex:4];
        [encoder setBytes:&seqK length:sizeof(seqK) atIndex:5];
        [encoder setBytes:&depth length:sizeof(depth) atIndex:6];
        [encoder setBytes:&valueDim length:sizeof(valueDim) atIndex:7];
        [encoder
            dispatchThreadgroups:MTLSizeMake(seqQ, (valueDim + 63) / 64, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_attention_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];
        return 0;
    }
}
