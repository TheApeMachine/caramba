#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#include "tensor.h"
#include <stdint.h>
#include <string.h>

static id<MTLDevice> gTensorDevice = nil;
static dispatch_once_t gTensorDeviceOnceToken;

int metal_tensor_init(void) {
    dispatch_once(&gTensorDeviceOnceToken, ^{
        @autoreleasepool {
            gTensorDevice = MTLCreateSystemDefaultDevice();
        }
    });

    return gTensorDevice ? METAL_TENSOR_OK : METAL_TENSOR_ERR_INIT;
}

void* metal_tensor_empty_float32(size_t n) {
    @autoreleasepool {
        if (n == 0) {
            return NULL;
        }

        if (n > SIZE_MAX / sizeof(float)) {
            return NULL;
        }

        if (metal_tensor_init() != METAL_TENSOR_OK) {
            return NULL;
        }

        size_t bytes = n * sizeof(float);
        id<MTLBuffer> buffer = [gTensorDevice newBufferWithLength:(NSUInteger)bytes
                                                          options:MTLResourceStorageModeShared];

        if (!buffer) {
            return NULL;
        }

        return (void*)buffer;
    }
}

void* metal_tensor_upload_float32(const float* src, size_t n) {
    @autoreleasepool {
        if (n == 0) {
            return NULL;
        }

        if (!src) {
            return NULL;
        }

        if (n > SIZE_MAX / sizeof(float)) {
            return NULL;
        }

        if (metal_tensor_init() != METAL_TENSOR_OK) {
            return NULL;
        }

        size_t bytes = n * sizeof(float);
        id<MTLBuffer> buffer = [gTensorDevice newBufferWithBytes:src
                                                          length:(NSUInteger)bytes
                                                         options:MTLResourceStorageModeShared];

        if (!buffer) {
            return NULL;
        }

        return (void*)buffer;
    }
}

int metal_tensor_download_float32(const void* handle, float* dst, size_t n) {
    @autoreleasepool {
        if (n == 0) {
            return METAL_TENSOR_OK;
        }

        if (!handle || !dst) {
            return METAL_TENSOR_ERR_NULL_PTR;
        }

        if (n > SIZE_MAX / sizeof(float)) {
            return METAL_TENSOR_ERR_OVERFLOW;
        }

        size_t bytes = n * sizeof(float);
        id<MTLBuffer> buffer = (id<MTLBuffer>)handle;

        if ([buffer length] < bytes) {
            return METAL_TENSOR_ERR_BOUNDS;
        }

        memcpy(dst, [buffer contents], bytes);

        return METAL_TENSOR_OK;
    }
}

size_t metal_tensor_get_size(const void* handle) {
    @autoreleasepool {
        if (!handle) {
            return 0;
        }

        id<MTLBuffer> buffer = (id<MTLBuffer>)handle;

        return (size_t)[buffer length];
    }
}

int metal_tensor_free(void* handle) {
    @autoreleasepool {
        if (!handle) {
            return METAL_TENSOR_OK;
        }

        id<MTLBuffer> buffer = (id<MTLBuffer>)handle;
        [buffer release];

        return METAL_TENSOR_OK;
    }
}
