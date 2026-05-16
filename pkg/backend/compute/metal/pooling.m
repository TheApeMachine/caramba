#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "pooling.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Static globals — initialized once via metal_pooling_init
// ---------------------------------------------------------------------------

static id<MTLDevice>               gPoolDevice         = nil;
static id<MTLCommandQueue>         gPoolQueue          = nil;
static id<MTLComputePipelineState> gPSO_maxPool        = nil;
static id<MTLComputePipelineState> gPSO_avgPool        = nil;
static id<MTLComputePipelineState> gPSO_adaptAvgPool   = nil;
static id<MTLComputePipelineState> gPSO_adaptMaxPool   = nil;

// ---------------------------------------------------------------------------

static id<MTLComputePipelineState> pool_make_pso(id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    id<MTLComputePipelineState> pso = [gPoolDevice newComputePipelineStateWithFunction:fn error:&err];
    return pso;
}

int metal_pooling_init(const char* metallib_path) {
    @autoreleasepool {
        gPoolDevice = MTLCreateSystemDefaultDevice();
        if (!gPoolDevice) return -1;

        gPoolQueue = [gPoolDevice newCommandQueue];
        if (!gPoolQueue) return -1;

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError* err   = nil;
        id<MTLLibrary> lib = [gPoolDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!lib) return -1;

        gPSO_maxPool      = pool_make_pso(lib, @"max_pool2d_kernel");
        gPSO_avgPool      = pool_make_pso(lib, @"avg_pool2d_kernel");
        gPSO_adaptAvgPool = pool_make_pso(lib, @"adaptive_avg_pool2d_kernel");
        gPSO_adaptMaxPool = pool_make_pso(lib, @"adaptive_max_pool2d_kernel");

        if (!gPSO_maxPool || !gPSO_avgPool ||
            !gPSO_adaptAvgPool || !gPSO_adaptMaxPool) return -1;

        return 0;
    }
}

// ---------------------------------------------------------------------------
// Generic dispatch helper: src buffer + dst buffer + params struct
// ---------------------------------------------------------------------------

static int pool_dispatch(
    id<MTLComputePipelineState> pso,
    const float* src, NSUInteger src_bytes,
    float*       dst, NSUInteger dst_bytes,
    const void*  params, NSUInteger params_bytes,
    int          grid_n)
{
    @autoreleasepool {
        vm_size_t page = getpagesize();

        id<MTLBuffer> bufSrc, bufDst;

        if (((uintptr_t)src % page) == 0) {
            bufSrc = [gPoolDevice newBufferWithBytesNoCopy:(void*)src
                                                   length:src_bytes
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        } else {
            bufSrc = [gPoolDevice newBufferWithBytes:src length:src_bytes
                                            options:MTLResourceStorageModeShared];
        }

        if (((uintptr_t)dst % page) == 0) {
            bufDst = [gPoolDevice newBufferWithBytesNoCopy:dst
                                                   length:dst_bytes
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        } else {
            bufDst = [gPoolDevice newBufferWithLength:dst_bytes
                                             options:MTLResourceStorageModeShared];
        }

        if (!bufSrc || !bufDst) return -1;

        id<MTLCommandBuffer>        cb  = [gPoolQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:params length:params_bytes atIndex:2];

        NSUInteger tw  = pso.threadExecutionWidth;
        NSUInteger tgs = tw;
        MTLSize threads     = MTLSizeMake((NSUInteger)grid_n, 1, 1);
        MTLSize threadgroup = MTLSizeMake(tgs, 1, 1);

        [enc dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (((uintptr_t)dst % page) != 0) {
            memcpy(dst, [bufDst contents], dst_bytes);
        }
        return 0;
    }
}

// ---------------------------------------------------------------------------
// MaxPool params struct (must match Metal shader layout)
// ---------------------------------------------------------------------------

typedef struct {
    int N, C, H, W;
    int kH, kW, sH, sW;
    int pH, pW, dH, dW;
    int Hout, Wout;
} MetalMaxPoolParams;

typedef struct {
    int N, C, H, W;
    int kH, kW, sH, sW;
    int pH, pW, dH, dW;
    int Hout, Wout;
    int count_include_pad;
    int divisor_override;
} MetalAvgPoolParams;

typedef struct {
    int N, C, H, W;
    int OutH, OutW;
} MetalAdaptivePoolParams;

// ---------------------------------------------------------------------------

static int pool_dispatch_tensor(
    id<MTLComputePipelineState> pso,
    const void* src,
    void* dst,
    const void* params,
    NSUInteger params_bytes,
    int grid_n)
{
    @autoreleasepool {
        if (!gPoolQueue || !pso || !src || !dst || !params || grid_n < 0) return -1;
        if (grid_n == 0) return 0;

        id<MTLBuffer> bufSrc = (__bridge id)((void*)src);
        id<MTLBuffer> bufDst = (__bridge id)dst;
        id<MTLCommandBuffer> cb = [gPoolQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:params length:params_bytes atIndex:2];

        MTLSize threads = MTLSizeMake((NSUInteger)grid_n, 1, 1);
        MTLSize threadgroup = MTLSizeMake(pso.threadExecutionWidth, 1, 1);

        [enc dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        return cb.error ? -1 : 0;
    }
}

int metal_max_pool2d(
    const float* src, float* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout)
{
    MetalMaxPoolParams p = { N,C,H,W, kH,kW,sH,sW, pH,pW,dH,dW, Hout,Wout };
    NSUInteger src_bytes = (NSUInteger)(N*C*H*W) * sizeof(float);
    NSUInteger dst_bytes = (NSUInteger)(N*C*Hout*Wout) * sizeof(float);
    return pool_dispatch(gPSO_maxPool,
        src, src_bytes, dst, dst_bytes,
        &p, sizeof(p), N*C*Hout*Wout);
}

int metal_avg_pool2d(
    const float* src, float* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout,
    int count_include_pad, int divisor_override)
{
    MetalAvgPoolParams p = { N,C,H,W, kH,kW,sH,sW, pH,pW,dH,dW, Hout,Wout,
                             count_include_pad, divisor_override };
    NSUInteger src_bytes = (NSUInteger)(N*C*H*W) * sizeof(float);
    NSUInteger dst_bytes = (NSUInteger)(N*C*Hout*Wout) * sizeof(float);
    return pool_dispatch(gPSO_avgPool,
        src, src_bytes, dst, dst_bytes,
        &p, sizeof(p), N*C*Hout*Wout);
}

int metal_adaptive_avg_pool2d(
    const float* src, float* dst,
    int N, int C, int H, int W,
    int OutH, int OutW)
{
    MetalAdaptivePoolParams p = { N,C,H,W, OutH,OutW };
    NSUInteger src_bytes = (NSUInteger)(N*C*H*W) * sizeof(float);
    NSUInteger dst_bytes = (NSUInteger)(N*C*OutH*OutW) * sizeof(float);
    return pool_dispatch(gPSO_adaptAvgPool,
        src, src_bytes, dst, dst_bytes,
        &p, sizeof(p), N*C*OutH*OutW);
}

int metal_adaptive_max_pool2d(
    const float* src, float* dst,
    int N, int C, int H, int W,
    int OutH, int OutW)
{
    MetalAdaptivePoolParams p = { N,C,H,W, OutH,OutW };
    NSUInteger src_bytes = (NSUInteger)(N*C*H*W) * sizeof(float);
    NSUInteger dst_bytes = (NSUInteger)(N*C*OutH*OutW) * sizeof(float);
    return pool_dispatch(gPSO_adaptMaxPool,
        src, src_bytes, dst, dst_bytes,
        &p, sizeof(p), N*C*OutH*OutW);
}

int metal_max_pool2d_tensor(
    const void* src, void* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout)
{
    MetalMaxPoolParams p = { N,C,H,W, kH,kW,sH,sW, pH,pW,dH,dW, Hout,Wout };
    return pool_dispatch_tensor(gPSO_maxPool, src, dst, &p, sizeof(p), N*C*Hout*Wout);
}

int metal_avg_pool2d_tensor(
    const void* src, void* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout,
    int count_include_pad, int divisor_override)
{
    MetalAvgPoolParams p = { N,C,H,W, kH,kW,sH,sW, pH,pW,dH,dW, Hout,Wout,
                             count_include_pad, divisor_override };
    return pool_dispatch_tensor(gPSO_avgPool, src, dst, &p, sizeof(p), N*C*Hout*Wout);
}

int metal_adaptive_avg_pool2d_tensor(
    const void* src, void* dst,
    int N, int C, int H, int W,
    int OutH, int OutW)
{
    MetalAdaptivePoolParams p = { N,C,H,W, OutH,OutW };
    return pool_dispatch_tensor(gPSO_adaptAvgPool, src, dst, &p, sizeof(p), N*C*OutH*OutW);
}

int metal_adaptive_max_pool2d_tensor(
    const void* src, void* dst,
    int N, int C, int H, int W,
    int OutH, int OutW)
{
    MetalAdaptivePoolParams p = { N,C,H,W, OutH,OutW };
    return pool_dispatch_tensor(gPSO_adaptMaxPool, src, dst, &p, sizeof(p), N*C*OutH*OutW);
}
