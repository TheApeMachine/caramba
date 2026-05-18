#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>
#include <string.h>

static const NSUInteger metalMathThreadCountObjC = 256;

static void metal_math_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_math_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_math_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

static int metal_math_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_math_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_math_status_set(status, -6, "unknown Metal math kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_math_status_set(status, -6, "Metal math kernel name overflow");
        return -6;
    }

    return 0;
}

static void metal_math_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer,
    id<MTLBuffer> validationBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] != MTLCommandBufferStatusCompleted) {
            NSError* error = [completedBuffer error];
            NSString* message = @"Metal math command buffer failed";

            if (error != nil) {
                message = [NSString stringWithFormat:@"%@: %@", message, [error localizedDescription]];
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
                    "Metal math kernel reported invalid scalar data"
                );
                return;
            }
        }

        metalCommandCompleted(completionToken, 0, "");
    }
}

static id<MTLBuffer> metal_math_validation_buffer(
    MetalDeviceRef contextRef,
    MetalStatus* status
) {
    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->device == NULL) {
        metal_math_status_set(status, -1, "invalid Metal context");
        return nil;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
    id<MTLBuffer> validationBuffer = [device
        newBufferWithLength:sizeof(uint32_t)
        options:MTLResourceStorageModeShared
    ];

    if (validationBuffer == nil) {
        metal_math_status_set(status, -9, "validation buffer allocation failed");
        return nil;
    }

    uint32_t zero = 0;
    memcpy([validationBuffer contents], &zero, sizeof(zero));
    return validationBuffer;
}

static int metal_math_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandBuffer>* commandBuffer,
    id<MTLComputePipelineState>* pipeline
) {
    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->queue == NULL || context->device == NULL) {
        metal_math_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
    *commandBuffer = [queue commandBuffer];

    if (*commandBuffer == nil) {
        metal_math_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

static int metal_math_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    id<MTLComputeCommandEncoder>* encoder
) {
    *encoder = [commandBuffer computeCommandEncoder];

    if (*encoder == nil) {
        metal_math_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [*encoder setComputePipelineState:pipeline];
    return 0;
}

int metal_dispatch_inv_sqrt_dim_scale(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef dimRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_math_status_clear(status);

        if (count == 0) {
            return 0;
        }

        if (inputRef == NULL || dimRef == NULL || outRef == NULL) {
            metal_math_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char kernelName[128];
        int nameCode = metal_math_kernel_name(
            kernelName, sizeof(kernelName), "inv_sqrt_dim_scale", elementDType, status
        );

        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLBuffer> validationBuffer = metal_math_validation_buffer(contextRef, status);

        if (validationBuffer == nil) {
            return -9;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_math_prepare(contextRef, kernelName, status, &commandBuffer, &pipeline);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_math_encoder(commandBuffer, pipeline, status, &encoder);

        if (encoderCode != 0) {
            return encoderCode;
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)dimRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
        [encoder setBytes:&count length:sizeof(count) atIndex:3];
        [encoder setBuffer:validationBuffer offset:0 atIndex:4];
        NSUInteger threadWidth = [pipeline threadExecutionWidth];
        if (threadWidth == 0) {
            threadWidth = 1;
        }

        [encoder
            dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_math_complete(completionToken, completedBuffer, validationBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_logsumexp(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_math_status_clear(status);

        if (rows == 0 || cols == 0) {
            return 0;
        }

        if (inputRef == NULL || outRef == NULL) {
            metal_math_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char kernelName[128];
        int nameCode = metal_math_kernel_name(
            kernelName, sizeof(kernelName), "logsumexp", elementDType, status
        );

        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_math_prepare(contextRef, kernelName, status, &commandBuffer, &pipeline);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_math_encoder(commandBuffer, pipeline, status, &encoder);

        if (encoderCode != 0) {
            return encoderCode;
        }

        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:2];
        [encoder
            dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(metalMathThreadCountObjC, 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_math_complete(completionToken, completedBuffer, nil);
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_outer(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_math_status_clear(status);

        if (rows == 0 || cols == 0) {
            return 0;
        }

        if (leftRef == NULL || rightRef == NULL || outRef == NULL) {
            metal_math_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char kernelName[128];
        int nameCode = metal_math_kernel_name(
            kernelName, sizeof(kernelName), "outer", elementDType, status
        );

        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLCommandBuffer> commandBuffer = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_math_prepare(contextRef, kernelName, status, &commandBuffer, &pipeline);

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_math_encoder(commandBuffer, pipeline, status, &encoder);

        if (encoderCode != 0) {
            return encoderCode;
        }

        uint32_t count = rows * cols;
        [encoder setBuffer:(__bridge id<MTLBuffer>)leftRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)rightRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
        [encoder setBytes:&count length:sizeof(count) atIndex:4];
        NSUInteger threadWidth = [pipeline threadExecutionWidth];
        if (threadWidth == 0) {
            threadWidth = 1;
        }

        [encoder
            dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_math_complete(completionToken, completedBuffer, nil);
        }];
        [commandBuffer commit];

        return 0;
    }
}
