#include "bridge_research_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

int metal_dispatch_pc_prediction(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef weightsRef,
    MetalBufferRef stateRef,
    MetalBufferRef outRef,
    uint32_t outCount,
    uint32_t inCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (weightsRef == NULL || stateRef == NULL || outRef == NULL) {
        metal_research_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_research_kernel_name(
        kernelName, sizeof(kernelName), "pc_prediction", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_research_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)outCount,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)weightsRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)stateRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&inCount length:sizeof(inCount) atIndex:3];
        }
    );
}

int metal_dispatch_pc_update_representation(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef weightsRef,
    MetalBufferRef stateRef,
    MetalBufferRef errorRef,
    MetalBufferRef outRef,
    uint32_t outCount,
    uint32_t inCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (weightsRef == NULL || stateRef == NULL || errorRef == NULL || outRef == NULL) {
        metal_research_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_research_kernel_name(
        kernelName, sizeof(kernelName), "pc_update_representation", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_research_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)inCount,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)weightsRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)stateRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)errorRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&outCount length:sizeof(outCount) atIndex:4];
            [encoder setBytes:&inCount length:sizeof(inCount) atIndex:5];
        }
    );
}

int metal_dispatch_pc_update_weights(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef weightsRef,
    MetalBufferRef stateRef,
    MetalBufferRef errorRef,
    MetalBufferRef outRef,
    uint32_t outCount,
    uint32_t inCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (weightsRef == NULL || stateRef == NULL || errorRef == NULL || outRef == NULL) {
        metal_research_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    char kernelName[128];
    int nameCode = metal_research_kernel_name(
        kernelName, sizeof(kernelName), "pc_update_weights", elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    uint32_t count = outCount * inCount;

    return metal_research_dispatch(
        contextRef,
        kernelName,
        (NSUInteger)count,
        completionToken,
        status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)weightsRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)stateRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)errorRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&inCount length:sizeof(inCount) atIndex:4];
            [encoder setBytes:&count length:sizeof(count) atIndex:5];
        }
    );
}
