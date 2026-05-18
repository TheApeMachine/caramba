#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static void metal_projection_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_projection_status_set(MetalStatus* status, int code, const char* message) {
    if (status == NULL) {
        return;
    }

    status->code = code;

    if (message == NULL) {
        status->message[0] = '\0';
        return;
    }

    snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "%s", message);
}

static const char* metal_projection_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

static int metal_projection_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_projection_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_projection_status_set(status, -6, "unknown Metal projection kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_projection_status_set(status, -6, "Metal projection kernel name overflow");
        return -6;
    }

    return 0;
}

static void metal_projection_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal projection command buffer failed";

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

static int metal_projection_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandQueue>* queue,
    id<MTLComputePipelineState>* pipeline
) {
    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->queue == NULL) {
        metal_projection_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *queue = (__bridge id<MTLCommandQueue>)context->queue;
    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

static int metal_projection_dispatch_2d(
    MetalDeviceRef contextRef,
    const char* kernelName,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status,
    void (^encode)(id<MTLComputeCommandEncoder> encoder)
) {
    @autoreleasepool {
        metal_projection_status_clear(status);

        id<MTLCommandQueue> queue = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_projection_prepare(
            contextRef, kernelName, status, &queue, &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_projection_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_projection_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [encoder setComputePipelineState:pipeline];
        encode(encoder);
        [encoder
            dispatchThreadgroups:MTLSizeMake((cols + 15) / 16, (rows + 15) / 16, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_projection_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

static int metal_projection_name_or_error(
    char* kernelName,
    size_t kernelNameBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    return metal_projection_kernel_name(
        kernelName,
        kernelNameBytes,
        operationName,
        elementDType,
        status
    );
}

int metal_dispatch_linear(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef weightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t inner,
    uint32_t outDim,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || weightRef == NULL || biasRef == NULL || outRef == NULL) {
        metal_projection_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_projection_name_or_error(
        kernelName, sizeof(kernelName), "linear", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_projection_dispatch_2d(
        contextRef,
        kernelName,
        batch,
        outDim,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)weightRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:4];
            [encoder setBytes:&inner length:sizeof(inner) atIndex:5];
            [encoder setBytes:&outDim length:sizeof(outDim) atIndex:6];
        }
    );
}

int metal_dispatch_fused_qkv(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef weightRef,
    MetalBufferRef biasRef,
    MetalBufferRef queryRef,
    MetalBufferRef keyRef,
    MetalBufferRef valueRef,
    uint32_t batch,
    uint32_t inner,
    uint32_t outDim,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || weightRef == NULL || biasRef == NULL ||
        queryRef == NULL || keyRef == NULL || valueRef == NULL) {
        metal_projection_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_projection_name_or_error(
        kernelName, sizeof(kernelName), "fused_qkv", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_projection_dispatch_2d(
        contextRef,
        kernelName,
        batch,
        outDim,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)weightRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)queryRef offset:0 atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)keyRef offset:0 atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)valueRef offset:0 atIndex:5];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:6];
            [encoder setBytes:&inner length:sizeof(inner) atIndex:7];
            [encoder setBytes:&outDim length:sizeof(outDim) atIndex:8];
        }
    );
}

int metal_dispatch_lora_merge(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef baseRef,
    MetalBufferRef loraARef,
    MetalBufferRef loraBRef,
    MetalBufferRef outRef,
    uint32_t outDim,
    uint32_t rank,
    uint32_t inner,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (baseRef == NULL || loraARef == NULL || loraBRef == NULL || outRef == NULL) {
        metal_projection_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_projection_name_or_error(
        kernelName, sizeof(kernelName), "lora_merge", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_projection_dispatch_2d(
        contextRef,
        kernelName,
        outDim,
        inner,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)baseRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)loraARef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)loraBRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&outDim length:sizeof(outDim) atIndex:4];
            [encoder setBytes:&rank length:sizeof(rank) atIndex:5];
            [encoder setBytes:&inner length:sizeof(inner) atIndex:6];
        }
    );
}

int metal_dispatch_lora_apply(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef baseRef,
    MetalBufferRef loraARef,
    MetalBufferRef loraBRef,
    MetalBufferRef inputRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t inner,
    uint32_t rank,
    uint32_t outDim,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_projection_status_clear(status);

        if (baseRef == NULL || loraARef == NULL || loraBRef == NULL ||
            inputRef == NULL || scratchRef == NULL || outRef == NULL) {
            metal_projection_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char stage1Name[128];
        char stage2Name[128];
        int stage1NameCode = metal_projection_name_or_error(
            stage1Name, sizeof(stage1Name), "lora_apply_stage1", elementDType, status
        );

        if (stage1NameCode != 0) {
            return stage1NameCode;
        }

        int stage2NameCode = metal_projection_name_or_error(
            stage2Name, sizeof(stage2Name), "lora_apply_stage2", elementDType, status
        );

        if (stage2NameCode != 0) {
            return stage2NameCode;
        }

        id<MTLCommandQueue> queue = nil;
        id<MTLComputePipelineState> stage1Pipeline = nil;
        int prepareCode = metal_projection_prepare(
            contextRef, stage1Name, status, &queue, &stage1Pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputePipelineState> stage2Pipeline = nil;
        prepareCode = metal_projection_prepare(contextRef, stage2Name, status, &queue, &stage2Pipeline);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_projection_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> stage1Encoder = [commandBuffer computeCommandEncoder];

        if (stage1Encoder == nil) {
            metal_projection_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [stage1Encoder setComputePipelineState:stage1Pipeline];
        [stage1Encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [stage1Encoder setBuffer:(__bridge id<MTLBuffer>)loraBRef offset:0 atIndex:1];
        [stage1Encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:2];
        [stage1Encoder setBytes:&batch length:sizeof(batch) atIndex:3];
        [stage1Encoder setBytes:&inner length:sizeof(inner) atIndex:4];
        [stage1Encoder setBytes:&rank length:sizeof(rank) atIndex:5];
        [stage1Encoder
            dispatchThreadgroups:MTLSizeMake((rank + 15) / 16, (batch + 15) / 16, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)
        ];
        [stage1Encoder endEncoding];

        id<MTLComputeCommandEncoder> stage2Encoder = [commandBuffer computeCommandEncoder];

        if (stage2Encoder == nil) {
            metal_projection_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [stage2Encoder setComputePipelineState:stage2Pipeline];
        [stage2Encoder setBuffer:(__bridge id<MTLBuffer>)baseRef offset:0 atIndex:0];
        [stage2Encoder setBuffer:(__bridge id<MTLBuffer>)loraARef offset:0 atIndex:1];
        [stage2Encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:2];
        [stage2Encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
        [stage2Encoder setBytes:&batch length:sizeof(batch) atIndex:4];
        [stage2Encoder setBytes:&rank length:sizeof(rank) atIndex:5];
        [stage2Encoder setBytes:&outDim length:sizeof(outDim) atIndex:6];
        [stage2Encoder
            dispatchThreadgroups:MTLSizeMake((outDim + 15) / 16, (batch + 15) / 16, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)
        ];
        [stage2Encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_projection_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
