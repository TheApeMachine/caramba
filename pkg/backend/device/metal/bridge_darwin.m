#include "bridge_darwin.h"

#include <CoreFoundation/CoreFoundation.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct MetalContext {
    void* device;
    void* queue;
    void* library;
    void* pipelineCache;
    void* pipelineLock;
} MetalContext;

static void metal_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_status_set(MetalStatus* status, int code, const char* message) {
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

static void metal_status_set_ns_error(
    MetalStatus* status,
    int code,
    NSString* operation,
    NSError* error
) {
    NSString* message = operation;

    if (error != nil) {
        message = [NSString
            stringWithFormat:@"%@: %@",
            operation,
            [error localizedDescription]
        ];
    }

    metal_status_set(status, code, [message UTF8String]);
}

static void metal_release_context(MetalContext* context) {
    if (context == NULL) {
        return;
    }

    if (context->pipelineLock != NULL) {
        CFRelease(context->pipelineLock);
        context->pipelineLock = NULL;
    }

    if (context->pipelineCache != NULL) {
        CFRelease(context->pipelineCache);
        context->pipelineCache = NULL;
    }

    if (context->library != NULL) {
        CFRelease(context->library);
        context->library = NULL;
    }

    if (context->queue != NULL) {
        CFRelease(context->queue);
        context->queue = NULL;
    }

    if (context->device != NULL) {
        CFRelease(context->device);
        context->device = NULL;
    }

    free(context);
}

static const char* metal_binary_float32_kernel_name(int operation) {
    switch (operation) {
    case MetalBinaryFloat32Add:
        return "add_float32";
    case MetalBinaryFloat32Sub:
        return "sub_float32";
    case MetalBinaryFloat32Mul:
        return "mul_float32";
    case MetalBinaryFloat32Div:
        return "div_float32";
    default:
        return NULL;
    }
}

static id<MTLComputePipelineState> metal_get_pipeline(
    MetalContext* context,
    const char* name,
    MetalStatus* status
) {
    if (context == NULL ||
        context->device == NULL ||
        context->library == NULL ||
        context->pipelineCache == NULL ||
        context->pipelineLock == NULL) {
        metal_status_set(status, -6, "invalid Metal pipeline context");
        return nil;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)context->library;
    NSMutableDictionary* pipelineCache =
        (__bridge NSMutableDictionary*)context->pipelineCache;
    NSLock* pipelineLock = (__bridge NSLock*)context->pipelineLock;
    NSString* functionName = [NSString stringWithUTF8String:name];

    [pipelineLock lock];
    id<MTLComputePipelineState> cachedPipeline = [pipelineCache objectForKey:functionName];
    [pipelineLock unlock];

    if (cachedPipeline != nil) {
        return cachedPipeline;
    }

    id<MTLFunction> function = [library newFunctionWithName:functionName];

    if (function == nil) {
        metal_status_set(status, -6, "newFunctionWithName returned nil");
        return nil;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];

    if (pipeline == nil) {
        metal_status_set_ns_error(status, -7, @"newComputePipelineStateWithFunction", error);
        return nil;
    }

    [pipelineLock lock];
    cachedPipeline = [pipelineCache objectForKey:functionName];

    if (cachedPipeline != nil) {
        [pipelineLock unlock];
        return cachedPipeline;
    }

    [pipelineCache setObject:pipeline forKey:functionName];
    [pipelineLock unlock];

    return pipeline;
}

MetalDeviceRef metal_open_default_device(
    const uint8_t* libraryBytes,
    long long libraryLength,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_status_clear(status);

        if (libraryBytes == NULL || libraryLength <= 0) {
            metal_status_set(status, -1, "empty Metal library");
            return NULL;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        if (device == nil) {
            metal_status_set(status, -2, "MTLCreateSystemDefaultDevice returned nil");
            return NULL;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        if (queue == nil) {
            metal_status_set(status, -3, "newCommandQueue returned nil");
            return NULL;
        }

        dispatch_data_t libraryData = dispatch_data_create(
            libraryBytes,
            (size_t)libraryLength,
            nil,
            DISPATCH_DATA_DESTRUCTOR_DEFAULT
        );

        if (libraryData == nil) {
            metal_status_set(status, -4, "dispatch_data_create returned nil");
            return NULL;
        }

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithData:libraryData error:&error];

        if (library == nil) {
            metal_status_set_ns_error(status, -5, @"newLibraryWithData", error);
            return NULL;
        }

        MetalContext* context = (MetalContext*)calloc(1, sizeof(MetalContext));

        if (context == NULL) {
            metal_status_set(status, -8, "calloc MetalContext failed");
            return NULL;
        }

        context->device = (__bridge_retained void*)device;
        context->queue = (__bridge_retained void*)queue;
        context->library = (__bridge_retained void*)library;
        context->pipelineCache = (__bridge_retained void*)[NSMutableDictionary dictionary];
        context->pipelineLock = (__bridge_retained void*)[[NSLock alloc] init];

        if (context->pipelineCache == NULL || context->pipelineLock == NULL) {
            metal_status_set(status, -9, "Metal pipeline cache initialization failed");
            metal_release_context(context);
            return NULL;
        }

        return context;
    }
}

long long metal_recommended_max_working_set(MetalDeviceRef contextRef) {
    MetalContext* context = (MetalContext*)contextRef;

    if (context == NULL || context->device == NULL) {
        return 0;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
    return (long long)[device recommendedMaxWorkingSetSize];
}

MetalBufferRef metal_buffer_new_shared(MetalDeviceRef contextRef, long long bytes) {
    @autoreleasepool {
        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->device == NULL || bytes <= 0) {
            return NULL;
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        id<MTLBuffer> buffer = [device
            newBufferWithLength:(NSUInteger)bytes
            options:MTLResourceStorageModeShared
        ];

        if (buffer == nil) {
            return NULL;
        }

        return (__bridge_retained void*)buffer;
    }
}

void metal_buffer_release(MetalBufferRef bufferRef) {
    if (bufferRef != NULL) {
        CFRelease(bufferRef);
    }
}

void* metal_buffer_contents(MetalBufferRef bufferRef) {
    if (bufferRef == NULL) {
        return NULL;
    }

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)bufferRef;
    return [buffer contents];
}

int metal_dispatch_binary_float32(
    MetalDeviceRef contextRef,
    int operation,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_status_clear(status);

        if (count == 0) {
            return 0;
        }

        MetalContext* context = (MetalContext*)contextRef;

        if (context == NULL || context->queue == NULL) {
            metal_status_set(status, -1, "invalid Metal context");
            return -1;
        }

        const char* kernelName = metal_binary_float32_kernel_name(operation);

        if (kernelName == NULL) {
            metal_status_set(status, -6, "unknown binary float32 operation");
            return -6;
        }

        if (leftRef == NULL || rightRef == NULL || outRef == NULL) {
            metal_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)context->queue;
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(context, kernelName, status);

        if (pipeline == nil) {
            return status != NULL && status->code != 0 ? status->code : -7;
        }

        id<MTLBuffer> left = (__bridge id<MTLBuffer>)leftRef;
        id<MTLBuffer> right = (__bridge id<MTLBuffer>)rightRef;
        id<MTLBuffer> out = (__bridge id<MTLBuffer>)outRef;

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:left offset:0 atIndex:0];
        [encoder setBuffer:right offset:0 atIndex:1];
        [encoder setBuffer:out offset:0 atIndex:2];
        [encoder setBytes:&count length:sizeof(count) atIndex:3];

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        NSUInteger vectorCount = (NSUInteger)((count + 3) / 4);
        MTLSize gridSize = MTLSizeMake(vectorCount, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(threadWidth, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            @autoreleasepool {
                if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
                    metalCommandCompleted(completionToken, 0, "");
                    return;
                }

                NSError* error = [completedBuffer error];
                NSString* message = @"Metal command buffer failed";

                if (error != nil) {
                    message = [NSString
                        stringWithFormat:@"%@: %@",
                        message,
                        [error localizedDescription]
                    ];
                }

                metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
            }
        }];

        [commandBuffer commit];

        return 0;
    }
}

void metal_device_release(MetalDeviceRef contextRef) {
    metal_release_context((MetalContext*)contextRef);
}
