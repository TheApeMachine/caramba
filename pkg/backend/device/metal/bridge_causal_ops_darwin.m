#include "bridge_causal_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

static int metal_causal_named_dispatch(
    MetalDeviceRef contextRef,
    int elementDType,
    const char* operationName,
    NSUInteger threadCount,
    uint64_t completionToken,
    MetalStatus* status,
    MetalCausalEncodeBlock encode
) {
    char kernelName[128];
    int nameCode = metal_causal_kernel_name(
        kernelName, sizeof(kernelName), operationName, elementDType, status
    );

    if (nameCode != 0) {
        return nameCode;
    }

    return metal_causal_dispatch(contextRef, kernelName, threadCount, completionToken, status, encode);
}

int metal_dispatch_backdoor_adjustment(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef conditionalRef,
    MetalBufferRef marginalRef,
    MetalBufferRef outRef,
    uint32_t xCount,
    uint32_t zCount,
    uint32_t yCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (conditionalRef == NULL || marginalRef == NULL || outRef == NULL) {
        metal_causal_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_causal_named_dispatch(
        contextRef, elementDType, "backdoor_adjustment", (NSUInteger)xCount * yCount,
        completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)conditionalRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)marginalRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&xCount length:sizeof(xCount) atIndex:3];
            [encoder setBytes:&zCount length:sizeof(zCount) atIndex:4];
            [encoder setBytes:&yCount length:sizeof(yCount) atIndex:5];
        }
    );
}

int metal_dispatch_frontdoor_adjustment(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef mediatorRef,
    MetalBufferRef outcomeRef,
    MetalBufferRef marginalRef,
    MetalBufferRef outRef,
    uint32_t xCount,
    uint32_t mCount,
    uint32_t yCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (mediatorRef == NULL || outcomeRef == NULL || marginalRef == NULL || outRef == NULL) {
        metal_causal_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_causal_named_dispatch(
        contextRef, elementDType, "frontdoor_adjustment", (NSUInteger)xCount * yCount,
        completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)mediatorRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outcomeRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)marginalRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:3];
            [encoder setBytes:&xCount length:sizeof(xCount) atIndex:4];
            [encoder setBytes:&mCount length:sizeof(mCount) atIndex:5];
            [encoder setBytes:&yCount length:sizeof(yCount) atIndex:6];
        }
    );
}

int metal_dispatch_do_intervene(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef adjacencyRef,
    MetalBufferRef intervenedRef,
    MetalBufferRef outRef,
    uint32_t nodeCount,
    uint32_t intervenedCount,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (adjacencyRef == NULL || intervenedRef == NULL || outRef == NULL) {
        metal_causal_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_causal_named_dispatch(
        contextRef, elementDType, "do_intervene", (NSUInteger)nodeCount * nodeCount,
        completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)adjacencyRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)intervenedRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&nodeCount length:sizeof(nodeCount) atIndex:3];
            [encoder setBytes:&intervenedCount length:sizeof(intervenedCount) atIndex:4];
        }
    );
}

int metal_dispatch_cate(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef treatedRef,
    MetalBufferRef controlRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (treatedRef == NULL || controlRef == NULL || outRef == NULL) {
        metal_causal_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_causal_named_dispatch(
        contextRef, elementDType, "cate", (NSUInteger)count, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)treatedRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)controlRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
        }
    );
}

int metal_dispatch_counterfactual(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef observedYRef,
    MetalBufferRef observedXRef,
    MetalBufferRef counterfactualXRef,
    MetalBufferRef slopeRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    if (observedYRef == NULL || observedXRef == NULL || counterfactualXRef == NULL ||
        slopeRef == NULL || outRef == NULL) {
        metal_causal_status_set(status, -2, "nil Metal buffer");
        return -2;
    }

    return metal_causal_named_dispatch(
        contextRef, elementDType, "counterfactual", (NSUInteger)count, completionToken, status,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)observedYRef offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)observedXRef offset:0 atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)counterfactualXRef offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)slopeRef offset:0 atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:4];
            [encoder setBytes:&count length:sizeof(count) atIndex:5];
        }
    );
}
