#include "bridge_vision_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

int metal_dispatch_conv1d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef weightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t inChannels,
    uint32_t inLength,
    uint32_t outChannels,
    uint32_t kernelLength,
    uint32_t outLength,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || weightRef == NULL || biasRef == NULL || outRef == NULL) {
        metal_vision_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_vision_kernel_name(
        kernelName, sizeof(kernelName), "conv1d", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    NSUInteger threadCount = (NSUInteger)batch * outChannels * outLength;
    return metal_vision_dispatch(
        contextRef, kernelName, threadCount, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)weightRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:4];
            [encoder setBytes:&inChannels length:sizeof(inChannels) atIndex:5];
            [encoder setBytes:&inLength length:sizeof(inLength) atIndex:6];
            [encoder setBytes:&outChannels length:sizeof(outChannels) atIndex:7];
            [encoder setBytes:&kernelLength length:sizeof(kernelLength) atIndex:8];
            [encoder setBytes:&outLength length:sizeof(outLength) atIndex:9];
        }
    );
}

int metal_dispatch_conv3d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef weightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t inChannels,
    uint32_t inDepth,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outChannels,
    uint32_t kernelDepth,
    uint32_t kernelHeight,
    uint32_t kernelWidth,
    uint32_t outDepth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || weightRef == NULL || biasRef == NULL || outRef == NULL) {
        metal_vision_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_vision_kernel_name(
        kernelName, sizeof(kernelName), "conv3d", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    NSUInteger threadCount = (NSUInteger)batch * outChannels * outDepth * outHeight * outWidth;
    return metal_vision_dispatch(
        contextRef, kernelName, threadCount, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)weightRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:4];
            [encoder setBytes:&inChannels length:sizeof(inChannels) atIndex:5];
            [encoder setBytes:&inDepth length:sizeof(inDepth) atIndex:6];
            [encoder setBytes:&inHeight length:sizeof(inHeight) atIndex:7];
            [encoder setBytes:&inWidth length:sizeof(inWidth) atIndex:8];
            [encoder setBytes:&outChannels length:sizeof(outChannels) atIndex:9];
            [encoder setBytes:&kernelDepth length:sizeof(kernelDepth) atIndex:10];
            [encoder setBytes:&kernelHeight length:sizeof(kernelHeight) atIndex:11];
            [encoder setBytes:&kernelWidth length:sizeof(kernelWidth) atIndex:12];
            [encoder setBytes:&outDepth length:sizeof(outDepth) atIndex:13];
            [encoder setBytes:&outHeight length:sizeof(outHeight) atIndex:14];
            [encoder setBytes:&outWidth length:sizeof(outWidth) atIndex:15];
        }
    );
}

int metal_dispatch_conv_transpose2d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef weightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t inChannels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outChannels,
    uint32_t kernelHeight,
    uint32_t kernelWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || weightRef == NULL || biasRef == NULL || outRef == NULL) {
        metal_vision_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_vision_kernel_name(
        kernelName, sizeof(kernelName), "conv_transpose2d", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    NSUInteger threadCount = (NSUInteger)batch * outChannels * outHeight * outWidth;
    return metal_vision_dispatch(
        contextRef, kernelName, threadCount, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)weightRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)biasRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:4];
            [encoder setBytes:&inChannels length:sizeof(inChannels) atIndex:5];
            [encoder setBytes:&inHeight length:sizeof(inHeight) atIndex:6];
            [encoder setBytes:&inWidth length:sizeof(inWidth) atIndex:7];
            [encoder setBytes:&outChannels length:sizeof(outChannels) atIndex:8];
            [encoder setBytes:&kernelHeight length:sizeof(kernelHeight) atIndex:9];
            [encoder setBytes:&kernelWidth length:sizeof(kernelWidth) atIndex:10];
            [encoder setBytes:&outHeight length:sizeof(outHeight) atIndex:11];
            [encoder setBytes:&outWidth length:sizeof(outWidth) atIndex:12];
        }
    );
}
