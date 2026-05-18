#include "bridge_transformer_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

int metal_dispatch_apply_mask(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef maskRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || maskRef == NULL || outRef == NULL) {
        metal_transformer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_transformer_kernel_name(
        kernelName, sizeof(kernelName), "apply_mask", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_transformer_dispatch(
        contextRef, kernelName, (NSUInteger)count, false, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            (void)validationBuffer;
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)maskRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
        }
    );
}

int metal_dispatch_causal_mask(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (outRef == NULL) {
        metal_transformer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_transformer_kernel_name(
        kernelName, sizeof(kernelName), "causal_mask", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_transformer_dispatch(
        contextRef, kernelName, (NSUInteger)rows * cols, false, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            (void)validationBuffer;
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:0];
            [encoder setBytes:&rows length:sizeof(rows) atIndex:1];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:2];
        }
    );
}

int metal_dispatch_alibi_bias(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef scoresRef,
    MetalBufferRef slopeRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (scoresRef == NULL || slopeRef == NULL || outRef == NULL) {
        metal_transformer_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_transformer_kernel_name(
        kernelName, sizeof(kernelName), "alibi_bias", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_transformer_dispatch(
        contextRef, kernelName, (NSUInteger)rows * cols, false, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> validationBuffer) {
            (void)validationBuffer;
            [encoder setBuffer:(__bridge id<MTLBuffer>)scoresRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)slopeRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        }
    );
}
