#include "bridge_research_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static const NSUInteger metalResearchThreadCount = 256;

static void metal_research_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_research_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_research_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

static const char* metal_research_operation_name(int operation) {
    switch (operation) {
    case 0: return "vsa_bind";
    case 1: return "vsa_bundle";
    case 2: return "vsa_permute";
    case 3: return "vsa_inverse_permute";
    case 4: return "pc_prediction_error";
    default: return NULL;
    }
}

int metal_research_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_research_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_research_status_set(status, -6, "unknown Metal research kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_research_status_set(status, -6, "Metal research kernel name overflow");
        return -6;
    }

    return 0;
}

static void metal_research_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal research command buffer failed";

        if (error != nil) {
            message = [NSString stringWithFormat:@"%@: %@", message, [error localizedDescription]];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

static int metal_research_prepare(
    MetalDeviceRef contextRef,
    const char* kernelName,
    MetalStatus* status,
    id<MTLComputePipelineState>* pipeline,
    id<MTLCommandBuffer>* commandBuffer
) {
    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->queue == NULL) {
        metal_research_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
    *commandBuffer = [queue commandBuffer];

    if (*commandBuffer == nil) {
        metal_research_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

static int metal_research_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    id<MTLComputeCommandEncoder>* encoder
) {
    *encoder = [commandBuffer computeCommandEncoder];

    if (*encoder == nil) {
        metal_research_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [*encoder setComputePipelineState:pipeline];
    return 0;
}

int metal_research_dispatch(
    MetalDeviceRef contextRef,
    const char* kernelName,
    NSUInteger count,
    uint64_t completionToken,
    MetalStatus* status,
    void (^encode)(id<MTLComputeCommandEncoder> encoder)
) {
    @autoreleasepool {
        metal_research_status_clear(status);

        if (count == 0) {
            return 0;
        }

        id<MTLComputePipelineState> pipeline = nil;
        id<MTLCommandBuffer> commandBuffer = nil;
        int prepareCode = metal_research_prepare(
            contextRef,
            kernelName,
            status,
            &pipeline,
            &commandBuffer
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        int encoderCode = metal_research_encoder(commandBuffer, pipeline, status, &encoder);

        if (encoderCode != 0) {
            return encoderCode;
        }

        encode(encoder);

        NSUInteger threadWidth = [pipeline threadExecutionWidth];
        if (threadWidth == 0) {
            threadWidth = metalResearchThreadCount;
        }

        [encoder
            dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_research_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_research_unary(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_research_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_research_kernel_name(
        kernelName,
        sizeof(kernelName),
        metal_research_operation_name(operation),
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_research_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)count,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&count length:sizeof(count) atIndex:2];
        }
    );
}

int metal_dispatch_research_binary(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (leftRef == NULL || rightRef == NULL || outRef == NULL) {
        metal_research_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_research_kernel_name(
        kernelName,
        sizeof(kernelName),
        metal_research_operation_name(operation),
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_research_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)count,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)leftRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)rightRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
        }
    );
}
