#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "tensor.h"
#include <string.h>

static id<MTLDevice> gTensorDevice = nil;

int metal_tensor_init(void) {
    @autoreleasepool {
        if (gTensorDevice) return 0;

        gTensorDevice = MTLCreateSystemDefaultDevice();
        if (!gTensorDevice) return -1;

        return 0;
    }
}

void* metal_tensor_empty_float32(int n) {
    @autoreleasepool {
        if (n == 0) return NULL;
        if (metal_tensor_init() != 0) return NULL;

        NSUInteger bytes = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> buffer = [gTensorDevice newBufferWithLength:bytes
                                                          options:MTLResourceStorageModeShared];
        if (!buffer) return NULL;

        return (void*)buffer;
    }
}

void* metal_tensor_upload_float32(const float* src, int n) {
    @autoreleasepool {
        if (n == 0) return NULL;
        if (!src) return NULL;
        if (metal_tensor_init() != 0) return NULL;

        NSUInteger bytes = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> buffer = [gTensorDevice newBufferWithBytes:src
                                                          length:bytes
                                                         options:MTLResourceStorageModeShared];
        if (!buffer) return NULL;

        return (void*)buffer;
    }
}

int metal_tensor_download_float32(const void* handle, float* dst, int n) {
    @autoreleasepool {
        if (n == 0) return 0;
        if (!handle || !dst) return -1;

        id<MTLBuffer> buffer = (id<MTLBuffer>)handle;
        NSUInteger bytes = (NSUInteger)n * sizeof(float);
        memcpy(dst, [buffer contents], bytes);

        return 0;
    }
}

int metal_tensor_free(void* handle) {
    @autoreleasepool {
        if (!handle) return 0;

        id<MTLBuffer> buffer = (id<MTLBuffer>)handle;
        [buffer release];

        return 0;
    }
}
