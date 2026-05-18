#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>
#include <string.h>

static const char* metal_fft_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

static bool metal_fft_is_power_of_two(uint32_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

static uint32_t metal_fft_log2(uint32_t value) {
    uint32_t bits = 0;
    while (value > 1) {
        value >>= 1;
        bits++;
    }

    return bits;
}

static void metal_fft_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_fft_status_set(MetalStatus* status, int code, const char* message) {
    if (status == NULL) {
        return;
    }

    status->code = code;
    snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "%s", message == NULL ? "" : message);
}

static void metal_fft_complete(uint64_t completionToken, id<MTLCommandBuffer> completedBuffer) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal FFT command buffer failed";
        if (error != nil) {
            message = [NSString stringWithFormat:@"%@: %@", message, [error localizedDescription]];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

static int metal_fft_kernel_name(
    char* out,
    size_t outBytes,
    int elementDType,
    const char* suffix,
    MetalStatus* status
) {
    const char* prefix = metal_fft_dtype_suffix(elementDType);
    if (prefix == NULL || suffix == NULL) {
        metal_fft_status_set(status, -6, "unknown Metal FFT kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", prefix, suffix);
    if (written <= 0 || (size_t)written >= outBytes) {
        metal_fft_status_set(status, -6, "Metal FFT kernel name overflow");
        return -6;
    }

    return 0;
}

static id<MTLComputePipelineState> metal_fft_pipeline(
    MetalContext* context,
    int elementDType,
    const char* suffix,
    MetalStatus* status
) {
    char kernelName[128];
    if (metal_fft_kernel_name(kernelName, sizeof(kernelName), elementDType, suffix, status) != 0) {
        return nil;
    }

    return metal_get_pipeline(context, kernelName, status);
}

static int metal_fft_encoder(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    MetalStatus* status,
    id<MTLComputeCommandEncoder>* encoder
) {
    *encoder = [commandBuffer computeCommandEncoder];
    if (*encoder == nil) {
        metal_fft_status_set(status, -4, "computeCommandEncoder returned nil");
        return -4;
    }

    [*encoder setComputePipelineState:pipeline];
    return 0;
}

static void metal_fft_dispatch(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    NSUInteger count
) {
    NSUInteger threadWidth = [pipeline threadExecutionWidth];
    if (threadWidth == 0) {
        threadWidth = 1;
    }

    [encoder
        dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
    ];
}

static int metal_fft_encode_naive(
    MetalContext* context,
    id<MTLCommandBuffer> commandBuffer,
    int elementDType,
    MetalBufferRef realInRef,
    MetalBufferRef imagInRef,
    MetalBufferRef realOutRef,
    MetalBufferRef imagOutRef,
    uint32_t count,
    bool inverse,
    MetalStatus* status
) {
    id<MTLComputePipelineState> pipeline = metal_fft_pipeline(context, elementDType, "dft_naive", status);
    if (pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    id<MTLComputeCommandEncoder> encoder = nil;
    int encoderCode = metal_fft_encoder(commandBuffer, pipeline, status, &encoder);
    if (encoderCode != 0) {
        return encoderCode;
    }

    uint32_t inverseValue = inverse ? 1u : 0u;
    [encoder setBuffer:(__bridge id<MTLBuffer>)realInRef offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)imagInRef offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)realOutRef offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)imagOutRef offset:0 atIndex:3];
    [encoder setBytes:&count length:sizeof(count) atIndex:4];
    [encoder setBytes:&inverseValue length:sizeof(inverseValue) atIndex:5];
    metal_fft_dispatch(encoder, pipeline, (NSUInteger)count);
    [encoder endEncoding];
    return 0;
}

static int metal_fft_encode_power2(
    MetalContext* context,
    id<MTLCommandBuffer> commandBuffer,
    int elementDType,
    MetalBufferRef realInRef,
    MetalBufferRef imagInRef,
    MetalBufferRef realOutRef,
    MetalBufferRef imagOutRef,
    uint32_t count,
    bool inverse,
    MetalStatus* status
) {
    id<MTLComputePipelineState> bitPipeline = metal_fft_pipeline(context, elementDType, "fft_bit_reverse", status);
    if (bitPipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    id<MTLComputeCommandEncoder> bitEncoder = nil;
    int bitCode = metal_fft_encoder(commandBuffer, bitPipeline, status, &bitEncoder);
    if (bitCode != 0) {
        return bitCode;
    }

    uint32_t bits = metal_fft_log2(count);
    [bitEncoder setBuffer:(__bridge id<MTLBuffer>)realInRef offset:0 atIndex:0];
    [bitEncoder setBuffer:(__bridge id<MTLBuffer>)imagInRef offset:0 atIndex:1];
    [bitEncoder setBuffer:(__bridge id<MTLBuffer>)realOutRef offset:0 atIndex:2];
    [bitEncoder setBuffer:(__bridge id<MTLBuffer>)imagOutRef offset:0 atIndex:3];
    [bitEncoder setBytes:&count length:sizeof(count) atIndex:4];
    [bitEncoder setBytes:&bits length:sizeof(bits) atIndex:5];
    metal_fft_dispatch(bitEncoder, bitPipeline, (NSUInteger)count);
    [bitEncoder endEncoding];

    id<MTLComputePipelineState> stagePipeline = metal_fft_pipeline(context, elementDType, "fft_stage", status);
    if (stagePipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    uint32_t inverseValue = inverse ? 1u : 0u;
    for (uint32_t length = 2; length <= count; length <<= 1) {
        id<MTLComputeCommandEncoder> stageEncoder = nil;
        int stageCode = metal_fft_encoder(commandBuffer, stagePipeline, status, &stageEncoder);
        if (stageCode != 0) {
            return stageCode;
        }

        [stageEncoder setBuffer:(__bridge id<MTLBuffer>)realOutRef offset:0 atIndex:0];
        [stageEncoder setBuffer:(__bridge id<MTLBuffer>)imagOutRef offset:0 atIndex:1];
        [stageEncoder setBytes:&length length:sizeof(length) atIndex:2];
        [stageEncoder setBytes:&inverseValue length:sizeof(inverseValue) atIndex:3];
        metal_fft_dispatch(stageEncoder, stagePipeline, (NSUInteger)(count / 2u));
        [stageEncoder endEncoding];
    }

    if (!inverse) {
        return 0;
    }

    id<MTLComputePipelineState> scalePipeline = metal_fft_pipeline(context, elementDType, "fft_scale", status);
    if (scalePipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    id<MTLComputeCommandEncoder> scaleEncoder = nil;
    int scaleCode = metal_fft_encoder(commandBuffer, scalePipeline, status, &scaleEncoder);
    if (scaleCode != 0) {
        return scaleCode;
    }

    [scaleEncoder setBuffer:(__bridge id<MTLBuffer>)realOutRef offset:0 atIndex:0];
    [scaleEncoder setBuffer:(__bridge id<MTLBuffer>)imagOutRef offset:0 atIndex:1];
    [scaleEncoder setBytes:&count length:sizeof(count) atIndex:2];
    metal_fft_dispatch(scaleEncoder, scalePipeline, (NSUInteger)count);
    [scaleEncoder endEncoding];
    return 0;
}

int metal_dispatch_fft1d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef realInRef,
    MetalBufferRef imagInRef,
    MetalBufferRef realOutRef,
    MetalBufferRef imagOutRef,
    uint32_t count,
    bool inverse,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_fft_status_clear(status);
        if (count == 0) {
            return 0;
        }

        if (realInRef == NULL || imagInRef == NULL || realOutRef == NULL || imagOutRef == NULL) {
            metal_fft_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        MetalContext* context = (MetalContext*)contextRef;
        if (context == NULL || context->queue == NULL || context->device == NULL) {
            metal_fft_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (commandBuffer == nil) {
            metal_fft_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        int encodeCode = 0;
        if (metal_fft_is_power_of_two(count)) {
            encodeCode = metal_fft_encode_power2(
                context, commandBuffer, elementDType, realInRef, imagInRef,
                realOutRef, imagOutRef, count, inverse, status
            );
        } else {
            encodeCode = metal_fft_encode_naive(
                context, commandBuffer, elementDType, realInRef, imagInRef,
                realOutRef, imagOutRef, count, inverse, status
            );
        }

        if (encodeCode != 0) {
            return encodeCode;
        }

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_fft_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
