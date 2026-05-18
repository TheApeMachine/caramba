#include "bridge_transformer_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>
#include <string.h>

static void metal_transformer_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_transformer_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_transformer_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

int metal_transformer_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_transformer_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_transformer_status_set(status, -6, "unknown Metal transformer kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_transformer_status_set(status, -6, "Metal transformer kernel name overflow");
        return -6;
    }

    return 0;
}

static void metal_transformer_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer,
    id<MTLBuffer> validationBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] != MTLCommandBufferStatusCompleted) {
            NSError* error = [completedBuffer error];
            NSString* message = @"Metal transformer command buffer failed";

            if (error != nil) {
                message = [NSString
                    stringWithFormat:@"%@: %@",
                    message,
                    [error localizedDescription]
                ];
            }

            metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
            return;
        }

        if (validationBuffer != nil) {
            uint32_t* validation = (uint32_t*)[validationBuffer contents];

            if (validation != NULL && validation[0] != 0) {
                metalCommandCompleted(
                    completionToken,
                    -8,
                    "Metal transformer kernel reported invalid index data"
                );
                return;
            }
        }

        metalCommandCompleted(completionToken, 0, "");
    }
}

static int metal_transformer_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    MetalContext** context,
    id<MTLCommandQueue>* queue,
    id<MTLComputePipelineState>* pipeline
) {
    *context = (MetalContext*)contextRef;

    if (*context == NULL || (*context)->queue == NULL || (*context)->device == NULL) {
        metal_transformer_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *queue = (__bridge id<MTLCommandQueue>)(*context)->queue;
    *pipeline = metal_get_pipeline(*context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

static id<MTLBuffer> metal_transformer_validation_buffer(
    MetalContext* context,
    MetalStatus* status
) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
    id<MTLBuffer> validationBuffer = [device
        newBufferWithLength:sizeof(uint32_t)
        options:MTLResourceStorageModeShared
    ];

    if (validationBuffer == nil) {
        metal_transformer_status_set(status, -9, "validation buffer allocation failed");
        return nil;
    }

    uint32_t zero = 0;
    memcpy([validationBuffer contents], &zero, sizeof(zero));
    return validationBuffer;
}

int metal_transformer_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    bool needsValidation,
    uint64_t completionToken,
    MetalStatus* status,
    MetalTransformerEncodeBlock encode
) {
    @autoreleasepool {
        metal_transformer_status_clear(status);

        if (threadCount == 0) {
            return 0;
        }

        MetalContext* context = NULL;
        id<MTLCommandQueue> queue = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_transformer_prepare(
            contextRef, kernelName, status, &context, &queue, &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLBuffer> validationBuffer = nil;
        if (needsValidation) {
            validationBuffer = metal_transformer_validation_buffer(context, status);

            if (validationBuffer == nil) {
                return -9;
            }
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_transformer_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_transformer_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [encoder setComputePipelineState:pipeline];
        encode(encoder, validationBuffer);

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        [encoder
            dispatchThreads:MTLSizeMake(threadCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_transformer_complete(completionToken, completedBuffer, validationBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_embedding_lookup(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef tableRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t vocab,
    uint32_t hidden,
    uint32_t indexCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (tableRef == NULL || indicesRef == NULL || outRef == NULL) {
        metal_transformer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_transformer_kernel_name(
        kernelName, sizeof(kernelName), "embedding_lookup", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_transformer_dispatch(
        contextRef, kernelName, (NSUInteger)indexCount * hidden, true, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)tableRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)indicesRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBuffer:validationBuffer offset:0 atIndex:3];
            [encoder setBytes:&vocab length:sizeof(vocab) atIndex:4];
            [encoder setBytes:&hidden length:sizeof(hidden) atIndex:5];
            [encoder setBytes:&indexCount length:sizeof(indexCount) atIndex:6];
        }
    );
}

int metal_dispatch_embedding_bag(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef tableRef,
    MetalBufferRef indicesRef,
    MetalBufferRef offsetsRef,
    MetalBufferRef outRef,
    uint32_t vocab,
    uint32_t hidden,
    uint32_t indexCount,
    uint32_t bagCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (tableRef == NULL || indicesRef == NULL || offsetsRef == NULL || outRef == NULL) {
        metal_transformer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_transformer_kernel_name(
        kernelName, sizeof(kernelName), "embedding_bag", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_transformer_dispatch(
        contextRef, kernelName, (NSUInteger)bagCount * hidden, true, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)tableRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)indicesRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)offsetsRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBuffer:validationBuffer offset:0 atIndex:4];
            [encoder setBytes:&vocab length:sizeof(vocab) atIndex:5];
            [encoder setBytes:&hidden length:sizeof(hidden) atIndex:6];
            [encoder setBytes:&indexCount length:sizeof(indexCount) atIndex:7];
            [encoder setBytes:&bagCount length:sizeof(bagCount) atIndex:8];
        }
    );
}
