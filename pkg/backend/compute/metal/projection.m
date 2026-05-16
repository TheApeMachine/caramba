#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "projection.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Static globals — initialised once via metal_projection_init
// ---------------------------------------------------------------------------

static id<MTLDevice>               gProjDevice     = nil;
static id<MTLCommandQueue>         gProjQueue      = nil;
static id<MTLComputePipelineState> gPSO_linear     = nil;
static id<MTLComputePipelineState> gPSO_fused_qkv  = nil;
static id<MTLComputePipelineState> gPSO_tied_emb   = nil;

// ---------------------------------------------------------------------------

static id<MTLComputePipelineState> make_proj_pso(id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    return [gProjDevice newComputePipelineStateWithFunction:fn error:&err];
}

int metal_projection_init(const char* metallib_path) {
    @autoreleasepool {
        gProjDevice = MTLCreateSystemDefaultDevice();
        if (!gProjDevice) return -1;

        gProjQueue = [gProjDevice newCommandQueue];
        if (!gProjQueue) return -1;

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError* err   = nil;
        id<MTLLibrary> lib = [gProjDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!lib) return -1;

        gPSO_linear    = make_proj_pso(lib, @"linear_kernel");
        gPSO_fused_qkv = make_proj_pso(lib, @"fused_qkv_kernel");
        gPSO_tied_emb  = make_proj_pso(lib, @"tied_embedding_kernel");

        if (!gPSO_linear || !gPSO_fused_qkv || !gPSO_tied_emb) return -1;
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Helper: dispatch a projection matmul kernel.
// Buffers layout:
//   [0] src (A)      — float*, src_n elements
//   [1] weight (W)   — float*, weight_n elements
//   [2] bias         — float*, n elements (may be NULL / zero-byte)
//   [3] dst (C)      — float*, dst_n elements
//   [4..6] dims      — uint M, K, N
//   [7] has_bias     — uint
// grid: (N, M)
// ---------------------------------------------------------------------------

static id<MTLBuffer> make_buf(id<MTLDevice> dev, const void* data, NSUInteger bytes, BOOL readonly) {
    if (!data || bytes == 0) {
        // Return a tiny zero-filled buffer for unused slots.
        return [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    }
    vm_size_t page = getpagesize();
    if (((uintptr_t)data % page) == 0) {
        return [dev newBufferWithBytesNoCopy:(void*)data length:bytes
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
    }
    return [dev newBufferWithBytes:data length:bytes
                           options:MTLResourceStorageModeShared];
}

static int dispatch_proj(
    id<MTLComputePipelineState> pso,
    const float* src,    int src_n,
    const float* weight, int weight_n,
    const float* bias,   int bias_n,
    float* dst,          int dst_n,
    unsigned int M, unsigned int K, unsigned int N,
    unsigned int has_bias)
{
    @autoreleasepool {
        NSUInteger src_bytes    = (NSUInteger)src_n    * sizeof(float);
        NSUInteger weight_bytes = (NSUInteger)weight_n * sizeof(float);
        NSUInteger bias_bytes   = bias ? (NSUInteger)bias_n * sizeof(float) : 0;
        NSUInteger dst_bytes    = (NSUInteger)dst_n    * sizeof(float);

        id<MTLBuffer> bufSrc    = make_buf(gProjDevice, src,    src_bytes,    YES);
        id<MTLBuffer> bufWeight = make_buf(gProjDevice, weight, weight_bytes, YES);
        id<MTLBuffer> bufBias   = make_buf(gProjDevice, bias,   bias_bytes,   YES);

        vm_size_t page = getpagesize();
        BOOL dst_nocopy = (((uintptr_t)dst % page) == 0);
        id<MTLBuffer> bufDst;
        if (dst_nocopy) {
            bufDst = [gProjDevice newBufferWithBytesNoCopy:dst length:dst_bytes
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
        } else {
            bufDst = [gProjDevice newBufferWithLength:dst_bytes
                                              options:MTLResourceStorageModeShared];
        }

        if (!bufSrc || !bufWeight || !bufBias || !bufDst) return -1;

        id<MTLCommandBuffer>         cb  = [gProjQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:bufSrc    offset:0 atIndex:0];
        [enc setBuffer:bufWeight offset:0 atIndex:1];
        [enc setBuffer:bufBias   offset:0 atIndex:2];
        [enc setBuffer:bufDst    offset:0 atIndex:3];
        [enc setBytes:&M length:sizeof(unsigned int) atIndex:4];
        [enc setBytes:&K length:sizeof(unsigned int) atIndex:5];
        [enc setBytes:&N length:sizeof(unsigned int) atIndex:6];
        [enc setBytes:&has_bias length:sizeof(unsigned int) atIndex:7];

        MTLSize threads    = MTLSizeMake((NSUInteger)N, (NSUInteger)M, 1);
        NSUInteger tw      = pso.threadExecutionWidth;
        MTLSize threadgroup = MTLSizeMake(tw, 1, 1);

        [enc dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (!dst_nocopy) {
            memcpy(dst, [bufDst contents], dst_bytes);
        }
        return 0;
    }
}

static int dispatch_proj_tensor(
    id<MTLComputePipelineState> pso,
    const void* src,
    const void* weight,
    const void* bias,
    void* dst,
    unsigned int M, unsigned int K, unsigned int N,
    unsigned int has_bias)
{
    @autoreleasepool {
        if (!gProjQueue || !pso || !src || !weight || !dst) return -1;
        if (M == 0 || K == 0 || N == 0) return -1;

        id<MTLBuffer> bufSrc = (__bridge id)((void*)src);
        id<MTLBuffer> bufWeight = (__bridge id)((void*)weight);
        id<MTLBuffer> bufBias = nil;
        BOOL releaseBias = NO;

        if (bias) {
            bufBias = (__bridge id)((void*)bias);
        } else {
            bufBias = [gProjDevice newBufferWithLength:4 options:MTLResourceStorageModeShared];
            releaseBias = YES;
        }

        id<MTLBuffer> bufDst = (__bridge id)dst;
        if (!bufSrc || !bufWeight || !bufBias || !bufDst) {
            if (releaseBias) [bufBias release];
            return -1;
        }

        id<MTLCommandBuffer> cb = [gProjQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) {
            if (releaseBias) [bufBias release];
            return -1;
        }

        [enc setComputePipelineState:pso];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufWeight offset:0 atIndex:1];
        [enc setBuffer:bufBias offset:0 atIndex:2];
        [enc setBuffer:bufDst offset:0 atIndex:3];
        [enc setBytes:&M length:sizeof(unsigned int) atIndex:4];
        [enc setBytes:&K length:sizeof(unsigned int) atIndex:5];
        [enc setBytes:&N length:sizeof(unsigned int) atIndex:6];
        [enc setBytes:&has_bias length:sizeof(unsigned int) atIndex:7];

        MTLSize threads = MTLSizeMake((NSUInteger)N, (NSUInteger)M, 1);
        MTLSize threadgroup = MTLSizeMake(pso.threadExecutionWidth, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        int rc = cb.error ? -1 : 0;
        if (releaseBias) [bufBias release];
        return rc;
    }
}

// ---------------------------------------------------------------------------

int metal_linear(const float* src, const float* weight, const float* bias,
                 float* dst, int M, int K, int N)
{
    unsigned int has_bias = (bias != NULL) ? 1u : 0u;
    return dispatch_proj(
        gPSO_linear,
        src,    M * K,
        weight, N * K,
        bias,   N,
        dst,    M * N,
        (unsigned int)M, (unsigned int)K, (unsigned int)N,
        has_bias);
}

int metal_fused_qkv(const float* src, const float* weight, const float* bias,
                    float* dst, int M, int K, int N)
{
    unsigned int has_bias = (bias != NULL) ? 1u : 0u;
    return dispatch_proj(
        gPSO_fused_qkv,
        src,    M * K,
        weight, N * K,
        bias,   N,
        dst,    M * N,
        (unsigned int)M, (unsigned int)K, (unsigned int)N,
        has_bias);
}

int metal_fused_qkv_tensor(
    const void* src,
    const void* weight,
    const void* bias,
    void*       dst,
    int         M,
    int         K,
    int         N)
{
    unsigned int has_bias = (bias != NULL) ? 1u : 0u;
    return dispatch_proj_tensor(
        gPSO_fused_qkv,
        src,
        weight,
        bias,
        dst,
        (unsigned int)M, (unsigned int)K, (unsigned int)N,
        has_bias);
}

int metal_tied_embedding(const float* src, const float* weight,
                         float* dst, int M, int D, int V)
{
    // Re-use linear kernel signature; no bias.
    return dispatch_proj(
        gPSO_tied_emb,
        src,    M * D,
        weight, V * D,
        NULL,   0,
        dst,    M * V,
        (unsigned int)M, (unsigned int)D, (unsigned int)V,
        0u);
}
