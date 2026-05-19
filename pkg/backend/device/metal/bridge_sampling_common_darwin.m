#include "bridge_sampling_private.h"

#include "_cgo_export.h"
#include <stdio.h>

static const NSUInteger metalSamplingThreadCountObjC = 256;

void metal_sampling_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

void metal_sampling_status_set(MetalStatus* status, int code, const char* message) {
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

static const char* metal_sampling_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

int metal_sampling_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    MetalStatus* status
) {
    const char* suffix = metal_sampling_dtype_suffix(elementDType);

    if (operationName == NULL || suffix == NULL) {
        metal_sampling_status_set(status, -6, "unknown Metal sampling kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, suffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_sampling_status_set(status, -6, "Metal sampling kernel name overflow");
        return -6;
    }

    return 0;
}

void metal_sampling_complete(uint64_t completionToken, id<MTLCommandBuffer> completedBuffer) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal sampling command buffer failed";

        if (error != nil) {
            message = [NSString stringWithFormat:@"%@: %@", message, [error localizedDescription]];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

int metal_sampling_prepare(
    MetalDeviceRef contextRef,
    MetalStatus* status,
    MetalContext** context,
    id<MTLCommandBuffer>* commandBuffer
) {
    *context = (MetalContext*)contextRef;

    if (*context == NULL || (*context)->queue == NULL || (*context)->device == NULL) {
        metal_sampling_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)(*context)->queue;
    *commandBuffer = [queue commandBuffer];

    if (*commandBuffer == nil) {
        metal_sampling_status_set(status, -3, "commandBuffer returned nil");
        return -3;
    }

    return 0;
}

int metal_sampling_pipeline(
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

int metal_sampling_encode_greedy(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef logitsRef,
    MetalBufferRef outRef,
    uint32_t count,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_sampling_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)logitsRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder
        dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metalSamplingThreadCountObjC, 1, 1)
    ];
    [encoder endEncoding];

    return 0;
}

int metal_sampling_encode_init(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalBufferRef logitsRef,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    uint32_t count,
    uint32_t paddedCount,
    MetalStatus* status
) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    if (encoder == nil) {
        metal_sampling_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)logitsRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)indicesRef offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
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
