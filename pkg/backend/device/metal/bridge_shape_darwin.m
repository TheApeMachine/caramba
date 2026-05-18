#include "bridge_shape_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

int metal_dispatch_copy_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t byteCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_shape_kernel_name(
        kernelName,
        sizeof(kernelName),
        "copy",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_shape_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)((byteCount + 15) / 16),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&byteCount length:sizeof(byteCount) atIndex:2];
        }
    );
}

int metal_dispatch_concat_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t leftBytes,
    uint32_t rightBytes,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (leftRef == NULL || rightRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    uint32_t totalBytes = leftBytes + rightBytes;
    char kernelName[128];
    int nameCode = metal_shape_kernel_name(
        kernelName,
        sizeof(kernelName),
        "concat",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_shape_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)((totalBytes + 15) / 16),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)leftRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)rightRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&leftBytes length:sizeof(leftBytes) atIndex:3];
            [encoder setBytes:&totalBytes length:sizeof(totalBytes) atIndex:4];
        }
    );
}

int metal_dispatch_split2_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    uint32_t leftBytes,
    uint32_t rightBytes,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || leftRef == NULL || rightRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    uint32_t totalBytes = leftBytes + rightBytes;
    char kernelName[128];
    int nameCode = metal_shape_kernel_name(
        kernelName,
        sizeof(kernelName),
        "split2",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_shape_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)((totalBytes + 15) / 16),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)leftRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)rightRef offset:0 atIndex:2];
            [encoder setBytes:&leftBytes length:sizeof(leftBytes) atIndex:3];
            [encoder setBytes:&totalBytes length:sizeof(totalBytes) atIndex:4];
        }
    );
}

int metal_dispatch_last_token_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t seq,
    uint32_t hiddenBytes,
    uint32_t outBytes,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_shape_kernel_name(
        kernelName,
        sizeof(kernelName),
        "last_token",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_shape_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)((outBytes + 15) / 16),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&seq length:sizeof(seq) atIndex:2];
            [encoder setBytes:&hiddenBytes length:sizeof(hiddenBytes) atIndex:3];
            [encoder setBytes:&outBytes length:sizeof(outBytes) atIndex:4];
        }
    );
}

int metal_dispatch_transpose2d_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_shape_kernel_name(
        kernelName,
        sizeof(kernelName),
        "transpose2d",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_shape_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)(rows * cols),
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
        }
    );
}

int metal_dispatch_upsample_nearest2d_bytes(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t channels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint32_t outElements,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (inputRef == NULL || outRef == NULL) {
        metal_shape_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_shape_kernel_name(
        kernelName,
        sizeof(kernelName),
        "upsample_nearest2d",
        elementDType,
        status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_shape_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)outElements,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
            [encoder setBytes:&channels length:sizeof(channels) atIndex:2];
            [encoder setBytes:&inHeight length:sizeof(inHeight) atIndex:3];
            [encoder setBytes:&inWidth length:sizeof(inWidth) atIndex:4];
            [encoder setBytes:&outHeight length:sizeof(outHeight) atIndex:5];
            [encoder setBytes:&outWidth length:sizeof(outWidth) atIndex:6];
            [encoder setBytes:&outElements length:sizeof(outElements) atIndex:7];
        }
    );
}
