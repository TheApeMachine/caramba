#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#include "tensor.h"
#include <stdint.h>
#include <string.h>

static id<MTLDevice> gTensorDevice = nil;
static id<MTLCommandQueue> gTensorQueue = nil;
static dispatch_once_t gTensorDeviceOnceToken;

int metal_tensor_init(void) {
    dispatch_once(&gTensorDeviceOnceToken, ^{
        @autoreleasepool {
            gTensorDevice = MTLCreateSystemDefaultDevice();
            gTensorQueue = [gTensorDevice newCommandQueue];
        }
    });

    return (gTensorDevice && gTensorQueue) ? METAL_TENSOR_OK : METAL_TENSOR_ERR_INIT;
}

void* metal_tensor_empty_float32(size_t n) {
    return metal_tensor_empty_float32_mode(n, 0);
}

static MTLResourceOptions metal_tensor_resource_options(int storage_mode) {
    if (storage_mode == 1) {
        return MTLResourceStorageModePrivate;
    }

    return MTLResourceStorageModeShared;
}

void* metal_tensor_empty_float32_mode(size_t n, int storage_mode) {
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
                                                          options:metal_tensor_resource_options(storage_mode)];

        if (!buffer) {
            return NULL;
        }

        return (void*)buffer;
    }
}

void* metal_tensor_upload_float32(const float* src, size_t n) {
    return metal_tensor_upload_float32_mode(src, n, 0);
}

static int metal_tensor_blit(id<MTLBuffer> source, id<MTLBuffer> destination, size_t bytes) {
    if (!gTensorQueue || !source || !destination) {
        return METAL_TENSOR_ERR_NULL_PTR;
    }

    id<MTLCommandBuffer> commandBuffer = [gTensorQueue commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
    [encoder copyFromBuffer:source
               sourceOffset:0
                   toBuffer:destination
          destinationOffset:0
                       size:(NSUInteger)bytes];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    return METAL_TENSOR_OK;
}

void* metal_tensor_upload_float32_mode(const float* src, size_t n, int storage_mode) {
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

        if (storage_mode == 1) {
            id<MTLBuffer> staging = [gTensorDevice newBufferWithBytes:src
                                                               length:(NSUInteger)bytes
                                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> buffer = [gTensorDevice newBufferWithLength:(NSUInteger)bytes
                                                              options:MTLResourceStorageModePrivate];

            if (!staging || !buffer) {
                [staging release];
                [buffer release];

                return NULL;
            }

            int rc = metal_tensor_blit(staging, buffer, bytes);
            [staging release];

            if (rc != METAL_TENSOR_OK) {
                [buffer release];

                return NULL;
            }

            return (void*)buffer;
        }

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

        if ([buffer storageMode] == MTLStorageModePrivate) {
            if (metal_tensor_init() != METAL_TENSOR_OK) {
                return METAL_TENSOR_ERR_INIT;
            }

            id<MTLBuffer> staging = [gTensorDevice newBufferWithLength:(NSUInteger)bytes
                                                                options:MTLResourceStorageModeShared];

            if (!staging) {
                return METAL_TENSOR_ERR_INIT;
            }

            int rc = metal_tensor_blit(buffer, staging, bytes);

            if (rc != METAL_TENSOR_OK) {
                [staging release];

                return rc;
            }

            memcpy(dst, [staging contents], bytes);
            [staging release];

            return METAL_TENSOR_OK;
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

int metal_tensor_get_storage_mode(const void* handle) {
    @autoreleasepool {
        if (!handle) {
            return -1;
        }

        id<MTLBuffer> buffer = (id<MTLBuffer>)handle;

        return (int)[buffer storageMode];
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
