#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>
#include <string.h>

static const NSUInteger metalLossThreadCount = 256;

static void metal_loss_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_loss_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_loss_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

static const char* metal_pair_loss_operation_name(int operation) {
    switch (operation) {
    case 0: return "mse_loss";
    case 1: return "mae_loss";
    case 2: return "huber_loss";
    case 3: return "binary_cross_entropy";
    case 4: return "kl_divergence";
    default: return NULL;
    }
}

static int metal_loss_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* phase,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_loss_dtype_suffix(elementDType);

    if (operationName == NULL || phase == NULL || suffix == NULL) {
        metal_loss_status_set(status, -6, "unknown Metal loss kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s_%s", operationName, suffix, phase);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_loss_status_set(status, -6, "Metal loss kernel name overflow");
        return -6;
    }

    return 0;
}

static int metal_loss_finalize_kernel_name(
    char* out,
    size_t outBytes,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_loss_dtype_suffix(elementDType);

    if (suffix == NULL) {
        metal_loss_status_set(status, -6, "unknown Metal loss finalize kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "loss_finalize_%s", suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_loss_status_set(status, -6, "Metal loss finalize kernel name overflow");
        return -6;
    }

    return 0;
}

static int metal_loss_prepare_pipeline(
    MetalContext* context,
    const char* kernelName,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline
) {
    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

static id<MTLBuffer> metal_loss_validation_buffer(
    MetalContext* context,
    MetalStatus* status
) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
    id<MTLBuffer> validationBuffer = [device
        newBufferWithLength:sizeof(uint32_t)
        options:MTLResourceStorageModeShared
    ];

    if (validationBuffer == nil) {
        metal_loss_status_set(status, -9, "validation buffer allocation failed");
        return nil;
    }

    uint32_t zero = 0;
    memcpy([validationBuffer contents], &zero, sizeof(zero));
    return validationBuffer;
}

static void metal_loss_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer,
    id<MTLBuffer> validationBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] != MTLCommandBufferStatusCompleted) {
            NSError* error = [completedBuffer error];
            NSString* message = @"Metal loss command buffer failed";

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
                    "Metal loss kernel reported invalid target data"
                );
                return;
            }
        }

        metalCommandCompleted(completionToken, 0, "");
    }
}

static int metal_loss_prepare_command(
    MetalDeviceRef contextRef,
    MetalStatus* status,
    MetalContext** context,
    id<MTLCommandQueue>* queue,
    id<MTLCommandBuffer>* commandBuffer
) {
    *context = (MetalContext*)contextRef;

    if (*context == NULL || (*context)->queue == NULL || (*context)->device == NULL) {
        metal_loss_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *queue = (__bridge id<MTLCommandQueue>)(*context)->queue;
    *commandBuffer = [*queue commandBuffer];

    if (*commandBuffer == nil) {
        metal_loss_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

static int metal_loss_encode_pair_partial(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef predictionsRef,
    MetalBufferRef targetsRef,
    MetalBufferRef scratchRef,
    uint32_t count,
    uint32_t partialCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_loss_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)predictionsRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)targetsRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)partialCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalLossThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

static int metal_loss_encode_cross_entropy_partial(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef logitsRef,
    MetalBufferRef targetsRef,
    MetalBufferRef scratchRef,
    id<MTLBuffer> validationBuffer,
    uint32_t batch,
    uint32_t classes,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_loss_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)logitsRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)targetsRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:2];
    [encoder setBuffer:validationBuffer offset:0 atIndex:3];
    [encoder setBytes:&batch length:sizeof(batch) atIndex:4];
    [encoder setBytes:&classes length:sizeof(classes) atIndex:5];
    [encoder
        dispatchThreadgroups:MTLSizeMake((NSUInteger)batch, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalLossThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

static int metal_loss_encode_finalize(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t partialCount,
    uint32_t denominator,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_loss_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scratchRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
    [encoder setBytes:&partialCount length:sizeof(partialCount) atIndex:2];
    [encoder setBytes:&denominator length:sizeof(denominator) atIndex:3];
    [encoder
        dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalLossThreadCount, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_dispatch_pair_loss(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef predictionsRef,
    MetalBufferRef targetsRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_loss_status_clear(status);

        if (count == 0 || partialCount == 0) {
            return 0;
        }

        if (predictionsRef == NULL || targetsRef == NULL || scratchRef == NULL || outRef == NULL) {
            metal_loss_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char partialName[128];
        char finalizeName[128];
        int partialNameCode = metal_loss_kernel_name(
            partialName,
            sizeof(partialName),
            metal_pair_loss_operation_name(operation),
            "partial",
            elementDType,
            status
        );

        if (partialNameCode != 0) {
            return partialNameCode;
        }

        int finalizeNameCode = metal_loss_finalize_kernel_name(
            finalizeName,
            sizeof(finalizeName),
            elementDType,
            status
        );

        if (finalizeNameCode != 0) {
            return finalizeNameCode;
        }

        MetalContext* context = NULL;
        id<MTLCommandQueue> queue = nil;
        id<MTLCommandBuffer> commandBuffer = nil;
        int commandCode = metal_loss_prepare_command(contextRef, status, &context, &queue, &commandBuffer);

        if (commandCode != 0) {
            return commandCode;
        }

        id<MTLComputePipelineState> partialPipeline = nil;
        int partialCode = metal_loss_prepare_pipeline(context, partialName, status, &partialPipeline);

        if (partialCode != 0) {
            return partialCode;
        }

        id<MTLComputePipelineState> finalizePipeline = nil;
        int finalizeCode = metal_loss_prepare_pipeline(context, finalizeName, status, &finalizePipeline);

        if (finalizeCode != 0) {
            return finalizeCode;
        }

        int encodePartialCode = metal_loss_encode_pair_partial(
            commandBuffer,
            partialPipeline,
            predictionsRef,
            targetsRef,
            scratchRef,
            count,
            partialCount,
            status
        );

        if (encodePartialCode != 0) {
            return encodePartialCode;
        }

        int encodeFinalizeCode = metal_loss_encode_finalize(
            commandBuffer,
            finalizePipeline,
            scratchRef,
            outRef,
            partialCount,
            count,
            status
        );

        if (encodeFinalizeCode != 0) {
            return encodeFinalizeCode;
        }

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_loss_complete(completionToken, completedBuffer, nil);
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_cross_entropy_loss(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef logitsRef,
    MetalBufferRef targetsRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t classes,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_loss_status_clear(status);

        if (batch == 0 || classes == 0) {
            return 0;
        }

        if (logitsRef == NULL || targetsRef == NULL || scratchRef == NULL || outRef == NULL) {
            metal_loss_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char partialName[128];
        char finalizeName[128];
        int partialNameCode = metal_loss_kernel_name(
            partialName,
            sizeof(partialName),
            "cross_entropy",
            "partial",
            elementDType,
            status
        );

        if (partialNameCode != 0) {
            return partialNameCode;
        }

        int finalizeNameCode = metal_loss_finalize_kernel_name(
            finalizeName,
            sizeof(finalizeName),
            elementDType,
            status
        );

        if (finalizeNameCode != 0) {
            return finalizeNameCode;
        }

        MetalContext* context = NULL;
        id<MTLCommandQueue> queue = nil;
        id<MTLCommandBuffer> commandBuffer = nil;
        int commandCode = metal_loss_prepare_command(contextRef, status, &context, &queue, &commandBuffer);

        if (commandCode != 0) {
            return commandCode;
        }

        id<MTLBuffer> validationBuffer = metal_loss_validation_buffer(context, status);

        if (validationBuffer == nil) {
            return -9;
        }

        id<MTLComputePipelineState> partialPipeline = nil;
        int partialCode = metal_loss_prepare_pipeline(context, partialName, status, &partialPipeline);

        if (partialCode != 0) {
            return partialCode;
        }

        id<MTLComputePipelineState> finalizePipeline = nil;
        int finalizeCode = metal_loss_prepare_pipeline(context, finalizeName, status, &finalizePipeline);

        if (finalizeCode != 0) {
            return finalizeCode;
        }

        int encodePartialCode = metal_loss_encode_cross_entropy_partial(
            commandBuffer,
            partialPipeline,
            logitsRef,
            targetsRef,
            scratchRef,
            validationBuffer,
            batch,
            classes,
            status
        );

        if (encodePartialCode != 0) {
            return encodePartialCode;
        }

        int encodeFinalizeCode = metal_loss_encode_finalize(
            commandBuffer,
            finalizePipeline,
            scratchRef,
            outRef,
            batch,
            batch,
            status
        );

        if (encodeFinalizeCode != 0) {
            return encodeFinalizeCode;
        }

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_loss_complete(completionToken, completedBuffer, validationBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
