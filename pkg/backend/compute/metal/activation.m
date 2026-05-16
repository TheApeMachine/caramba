#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#include <limits.h>
#include "activation.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Static globals — initialized once via metal_init
// ---------------------------------------------------------------------------

static id<MTLDevice>              gDevice       = nil;
static id<MTLCommandQueue>        gQueue        = nil;
static id<MTLComputePipelineState> gPSO_relu    = nil;
static id<MTLComputePipelineState> gPSO_leaky   = nil;
static id<MTLComputePipelineState> gPSO_gelu    = nil;
static id<MTLComputePipelineState> gPSO_tanh    = nil;
static id<MTLComputePipelineState> gPSO_sigmoid = nil;
static id<MTLComputePipelineState> gPSO_swish   = nil;
static id<MTLComputePipelineState> gPSO_selu    = nil;
static id<MTLComputePipelineState> gPSO_swiglu  = nil;

// ---------------------------------------------------------------------------

static id<MTLComputePipelineState> make_pso(id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    id<MTLComputePipelineState> pso = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return pso;
}

int metal_init(const char* metallib_path) {
    @autoreleasepool {
        gDevice = MTLCreateSystemDefaultDevice();
        if (!gDevice) return -1;

        gQueue = [gDevice newCommandQueue];
        if (!gQueue) return -1;

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError* err   = nil;
        id<MTLLibrary> lib = [gDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!lib) return -1;

        gPSO_relu    = make_pso(lib, @"relu_forward");
        gPSO_leaky   = make_pso(lib, @"leaky_relu_forward");
        gPSO_gelu    = make_pso(lib, @"gelu_forward");
        gPSO_tanh    = make_pso(lib, @"tanh_forward");
        gPSO_sigmoid = make_pso(lib, @"sigmoid_forward");
        gPSO_swish   = make_pso(lib, @"swish_forward");
        gPSO_selu    = make_pso(lib, @"selu_forward");
        gPSO_swiglu  = make_pso(lib, @"swiglu_forward");

        if (!gPSO_relu || !gPSO_leaky || !gPSO_gelu || !gPSO_tanh ||
            !gPSO_sigmoid || !gPSO_swish || !gPSO_selu || !gPSO_swiglu) return -1;

        return 0;
    }
}

// ---------------------------------------------------------------------------
// Helper: dispatch a kernel with two buffers (src, dst) and an optional
// scalar stored in a third binding.  scalar_bytes==0 means no third buffer.
// src_n is the number of float elements in src; dst_n in dst; grid_n is the
// thread grid size (== output length).
// ---------------------------------------------------------------------------

static int dispatch_2buf(
    id<MTLComputePipelineState> pso,
    const float* src, int src_n,
    float* dst,       int dst_n,
    const void* scalar, NSUInteger scalar_bytes,
    int grid_n)
{
    @autoreleasepool {
        // Use newBufferWithBytesNoCopy — no copy, GPU reads directly from
        // the Go-allocated float32 staging memory.
        // The page-size alignment requirement: Apple's Metal requires the
        // base pointer to be page-aligned for NoCopy.  We fall back to a
        // regular buffer if needed (caller must ensure alignment or accept copy).
        vm_size_t page = getpagesize();

        NSUInteger src_bytes = (NSUInteger)src_n * sizeof(float);
        NSUInteger dst_bytes = (NSUInteger)dst_n * sizeof(float);

        id<MTLBuffer> bufSrc, bufDst;

        if (((uintptr_t)src % page) == 0) {
            bufSrc = [gDevice newBufferWithBytesNoCopy:(void*)src
                                               length:src_bytes
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
        } else {
            bufSrc = [gDevice newBufferWithBytes:src length:src_bytes
                                        options:MTLResourceStorageModeShared];
        }

        if (((uintptr_t)dst % page) == 0) {
            bufDst = [gDevice newBufferWithBytesNoCopy:dst
                                               length:dst_bytes
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
        } else {
            bufDst = [gDevice newBufferWithLength:dst_bytes
                                         options:MTLResourceStorageModeShared];
        }

        if (!bufSrc || !bufDst) {
            [bufSrc release];
            [bufDst release];
            return -1;
        }

        id<MTLCommandBuffer>        cb  = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];

        if (scalar && scalar_bytes > 0) {
            [enc setBytes:scalar length:scalar_bytes atIndex:2];
        }

        NSUInteger tw  = pso.threadExecutionWidth;
        NSUInteger tgs = tw;
        MTLSize threads    = MTLSizeMake((NSUInteger)grid_n, 1, 1);
        MTLSize threadgroup = MTLSizeMake(tgs, 1, 1);

        [enc dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        // If dst was not page-aligned, copy back from the MTLBuffer.
        if (((uintptr_t)dst % page) != 0) {
            memcpy(dst, [bufDst contents], dst_bytes);
        }

        [bufSrc release];
        [bufDst release];
        return 0;
    }
}

static int dispatch_tensor_2buf(
    id<MTLComputePipelineState> pso,
    void* src,
    void* dst,
    const void* scalar, NSUInteger scalar_bytes,
    int grid_n,
    NSUInteger min_src_bytes,
    NSUInteger min_dst_bytes)
{
    @autoreleasepool {
        if (!gQueue || !pso || !src || !dst) return -1;

        if (grid_n <= 0) {
            return 0;
        }

        id srcObj = (__bridge id)(void*)src;
        id dstObj = (__bridge id)(void*)dst;

        if (![srcObj respondsToSelector:@selector(length)] ||
            ![dstObj respondsToSelector:@selector(length)]) {
            return -1;
        }

        NSUInteger srcLen = [(id<MTLBuffer>)srcObj length];
        NSUInteger dstLen = [(id<MTLBuffer>)dstObj length];

        if (srcLen < min_src_bytes || dstLen < min_dst_bytes) {
            return -1;
        }

        id<MTLBuffer> bufSrc = (id<MTLBuffer>)srcObj;
        id<MTLBuffer> bufDst = (id<MTLBuffer>)dstObj;

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];

        if (!cb) return -1;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        if (!enc) return -1;

        [enc setComputePipelineState:pso];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];

        if (scalar && scalar_bytes > 0) {
            [enc setBytes:scalar length:scalar_bytes atIndex:2];
        }

        NSUInteger tw = pso.threadExecutionWidth;
        MTLSize threads = MTLSizeMake((NSUInteger)grid_n, 1, 1);
        MTLSize threadgroup = MTLSizeMake(tw, 1, 1);

        [enc dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [enc endEncoding];

        __block int gpuOk = 1;
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);

        if (!sem) {
            return -1;
        }

        [cb addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            if (completed.status != MTLCommandBufferStatusCompleted) {
                gpuOk = 0;
            }

            dispatch_semaphore_signal(sem);
        }];
        [cb commit];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

        return gpuOk ? 0 : -1;
    }
}

// ---------------------------------------------------------------------------

int metal_relu(const float* src, float* dst, int n) {
    return dispatch_2buf(gPSO_relu, src, n, dst, n, NULL, 0, n);
}

int metal_leaky_relu(const float* src, float* dst, float alpha, int n) {
    return dispatch_2buf(gPSO_leaky, src, n, dst, n, &alpha, sizeof(float), n);
}

int metal_gelu(const float* src, float* dst, int n) {
    return dispatch_2buf(gPSO_gelu, src, n, dst, n, NULL, 0, n);
}

int metal_tanh(const float* src, float* dst, int n) {
    return dispatch_2buf(gPSO_tanh, src, n, dst, n, NULL, 0, n);
}

int metal_sigmoid(const float* src, float* dst, int n) {
    return dispatch_2buf(gPSO_sigmoid, src, n, dst, n, NULL, 0, n);
}

int metal_swish(const float* src, float* dst, int n) {
    return dispatch_2buf(gPSO_swish, src, n, dst, n, NULL, 0, n);
}

int metal_selu(const float* src, float* dst, int n) {
    return dispatch_2buf(gPSO_selu, src, n, dst, n, NULL, 0, n);
}

int metal_swiglu(const float* src, float* dst, int n) {
    unsigned int shape[2] = {(unsigned int)n, (unsigned int)(2 * n)};
    // src has 2*n elements (gates then values); dst has n elements.
    return dispatch_2buf(gPSO_swiglu, src, 2*n, dst, n, shape, sizeof(shape), n);
}

int metal_relu_tensor(const void* src, void* dst, int n) {
    if (n < 0) return -1;

    if (n == 0) return 0;

    if ((NSUInteger)n > ULONG_MAX / sizeof(float)) return -1;

    NSUInteger minB = (NSUInteger)n * sizeof(float);

    return dispatch_tensor_2buf(gPSO_relu, (void*)src, dst, NULL, 0, n, minB, minB);
}

int metal_leaky_relu_tensor(const void* src, void* dst, float alpha, int n) {
    if (n < 0) return -1;

    if (n == 0) return 0;

    if ((NSUInteger)n > ULONG_MAX / sizeof(float)) return -1;

    NSUInteger minB = (NSUInteger)n * sizeof(float);

    return dispatch_tensor_2buf(gPSO_leaky, (void*)src, dst, &alpha, sizeof(float), n, minB, minB);
}

int metal_gelu_tensor(const void* src, void* dst, int n) {
    if (n < 0) return -1;

    if (n == 0) return 0;

    if ((NSUInteger)n > ULONG_MAX / sizeof(float)) return -1;

    NSUInteger minB = (NSUInteger)n * sizeof(float);

    return dispatch_tensor_2buf(gPSO_gelu, (void*)src, dst, NULL, 0, n, minB, minB);
}

int metal_tanh_tensor(const void* src, void* dst, int n) {
    if (n < 0) return -1;

    if (n == 0) return 0;

    if ((NSUInteger)n > ULONG_MAX / sizeof(float)) return -1;

    NSUInteger minB = (NSUInteger)n * sizeof(float);

    return dispatch_tensor_2buf(gPSO_tanh, (void*)src, dst, NULL, 0, n, minB, minB);
}

int metal_sigmoid_tensor(const void* src, void* dst, int n) {
    if (n < 0) return -1;

    if (n == 0) return 0;

    if ((NSUInteger)n > ULONG_MAX / sizeof(float)) return -1;

    NSUInteger minB = (NSUInteger)n * sizeof(float);

    return dispatch_tensor_2buf(gPSO_sigmoid, (void*)src, dst, NULL, 0, n, minB, minB);
}

int metal_swish_tensor(const void* src, void* dst, int n) {
    if (n < 0) return -1;

    if (n == 0) return 0;

    if ((NSUInteger)n > ULONG_MAX / sizeof(float)) return -1;

    NSUInteger minB = (NSUInteger)n * sizeof(float);

    return dispatch_tensor_2buf(gPSO_swish, (void*)src, dst, NULL, 0, n, minB, minB);
}

int metal_selu_tensor(const void* src, void* dst, int n) {
    if (n < 0) return -1;

    if (n == 0) return 0;

    if ((NSUInteger)n > ULONG_MAX / sizeof(float)) return -1;

    NSUInteger minB = (NSUInteger)n * sizeof(float);

    return dispatch_tensor_2buf(gPSO_selu, (void*)src, dst, NULL, 0, n, minB, minB);
}

int metal_swiglu_tensor(const void* src, void* dst, int n, int input_width) {
    if (n <= 0) return -1;

    if (input_width <= 0 || input_width % 2 != 0) return -1;

    int output_width = input_width / 2;

    if (output_width <= 0 || n % output_width != 0) return -1;

    int rows = n / output_width;

    if (rows > INT_MAX / input_width) return -1;

    int src_n = rows * input_width;

    if ((NSUInteger)src_n > ULONG_MAX / sizeof(float)) return -1;
    if ((NSUInteger)n > ULONG_MAX / sizeof(float)) return -1;

    NSUInteger minSrc = (NSUInteger)src_n * sizeof(float);
    NSUInteger minDst = (NSUInteger)n * sizeof(float);
    unsigned int shape[2] = {(unsigned int)n, (unsigned int)input_width};

    return dispatch_tensor_2buf(
        gPSO_swiglu, (void*)src, dst, shape, sizeof(shape), n, minSrc, minDst);
}
