#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "attention.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Static globals — initialized once via metal_init_attention
// ---------------------------------------------------------------------------

static id<MTLDevice>               gAttnDevice = nil;
static id<MTLCommandQueue>         gAttnQueue  = nil;
static id<MTLComputePipelineState> gPSO_sdpa   = nil;
static id<MTLComputePipelineState> gPSO_mqa    = nil;
static id<MTLComputePipelineState> gPSO_gqa    = nil;
static id<MTLComputePipelineState> gPSO_sw     = nil;

// ---------------------------------------------------------------------------

static id<MTLComputePipelineState> attn_make_pso(id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    id<MTLComputePipelineState> pso = [gAttnDevice newComputePipelineStateWithFunction:fn error:&err];
    return pso;
}

int metal_init_attention(const char* metallib_path) {
    @autoreleasepool {
        gAttnDevice = MTLCreateSystemDefaultDevice();
        if (!gAttnDevice) return -1;

        gAttnQueue = [gAttnDevice newCommandQueue];
        if (!gAttnQueue) return -1;

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError* err   = nil;
        id<MTLLibrary> lib = [gAttnDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!lib) return -1;

        gPSO_sdpa = attn_make_pso(lib, @"sdpa_forward");
        gPSO_mqa  = attn_make_pso(lib, @"mqa_forward");
        gPSO_gqa  = attn_make_pso(lib, @"gqa_forward");
        gPSO_sw   = attn_make_pso(lib, @"sliding_window_forward");

        if (!gPSO_sdpa || !gPSO_mqa || !gPSO_gqa || !gPSO_sw) return -1;

        return 0;
    }
}

// ---------------------------------------------------------------------------
// Helper: create a Metal buffer from a host float pointer, respecting page
// alignment for zero-copy.
// ---------------------------------------------------------------------------

static id<MTLBuffer> make_buf_ro(id<MTLDevice> dev, const float* ptr, NSUInteger bytes) {
    vm_size_t page = getpagesize();
    if (((uintptr_t)ptr % page) == 0) {
        return [dev newBufferWithBytesNoCopy:(void*)ptr
                                      length:bytes
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
    }
    return [dev newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> make_buf_rw(id<MTLDevice> dev, float* ptr, NSUInteger bytes) {
    vm_size_t page = getpagesize();
    if (((uintptr_t)ptr % page) == 0) {
        return [dev newBufferWithBytesNoCopy:ptr
                                      length:bytes
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
    }
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

static void copy_back_if_needed(id<MTLBuffer> buf, float* dst, NSUInteger bytes) {
    vm_size_t page = getpagesize();
    if (((uintptr_t)dst % page) != 0) {
        memcpy(dst, [buf contents], bytes);
    }
}

// ---------------------------------------------------------------------------
// SDPA dispatch
// ---------------------------------------------------------------------------

int metal_sdpa(const float* q, const float* k, const float* v, float* out,
               int batch, int num_heads, int seq_len, int head_dim)
{
    @autoreleasepool {
        int total_heads = batch * num_heads;
        NSUInteger qkv_bytes = (NSUInteger)total_heads * seq_len * head_dim * sizeof(float);
        NSUInteger out_bytes = qkv_bytes;

        id<MTLBuffer> bufQ   = make_buf_ro(gAttnDevice, q, qkv_bytes);
        id<MTLBuffer> bufK   = make_buf_ro(gAttnDevice, k, qkv_bytes);
        id<MTLBuffer> bufV   = make_buf_ro(gAttnDevice, v, qkv_bytes);
        id<MTLBuffer> bufOut = make_buf_rw(gAttnDevice, out, out_bytes);
        if (!bufQ || !bufK || !bufV || !bufOut) return -1;

        // Threadgroup shared memory: seq_len * seq_len floats
        NSUInteger smem_bytes = (NSUInteger)seq_len * seq_len * sizeof(float);

        id<MTLCommandBuffer>        cb  = [gAttnQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:gPSO_sdpa];
        [enc setBuffer:bufQ   offset:0 atIndex:0];
        [enc setBuffer:bufK   offset:0 atIndex:1];
        [enc setBuffer:bufV   offset:0 atIndex:2];
        [enc setBuffer:bufOut offset:0 atIndex:3];
        [enc setBytes:&seq_len  length:sizeof(int) atIndex:4];
        [enc setBytes:&head_dim length:sizeof(int) atIndex:5];
        [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];

        MTLSize threadgroups    = MTLSizeMake((NSUInteger)total_heads, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake((NSUInteger)seq_len, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        copy_back_if_needed(bufOut, out, out_bytes);
        return 0;
    }
}

// ---------------------------------------------------------------------------
// MQA dispatch
// ---------------------------------------------------------------------------

int metal_mqa(const float* q, const float* k, const float* v, float* out,
              int batch, int num_heads, int seq_len, int head_dim)
{
    @autoreleasepool {
        int total_q_heads  = batch * num_heads;
        int total_kv_heads = batch * 1;

        NSUInteger q_bytes   = (NSUInteger)total_q_heads  * seq_len * head_dim * sizeof(float);
        NSUInteger kv_bytes  = (NSUInteger)total_kv_heads * seq_len * head_dim * sizeof(float);
        NSUInteger out_bytes = q_bytes;

        id<MTLBuffer> bufQ   = make_buf_ro(gAttnDevice, q, q_bytes);
        id<MTLBuffer> bufK   = make_buf_ro(gAttnDevice, k, kv_bytes);
        id<MTLBuffer> bufV   = make_buf_ro(gAttnDevice, v, kv_bytes);
        id<MTLBuffer> bufOut = make_buf_rw(gAttnDevice, out, out_bytes);
        if (!bufQ || !bufK || !bufV || !bufOut) return -1;

        NSUInteger smem_bytes = (NSUInteger)seq_len * seq_len * sizeof(float);

        id<MTLCommandBuffer>        cb  = [gAttnQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:gPSO_mqa];
        [enc setBuffer:bufQ   offset:0 atIndex:0];
        [enc setBuffer:bufK   offset:0 atIndex:1];
        [enc setBuffer:bufV   offset:0 atIndex:2];
        [enc setBuffer:bufOut offset:0 atIndex:3];
        [enc setBytes:&seq_len  length:sizeof(int) atIndex:4];
        [enc setBytes:&head_dim length:sizeof(int) atIndex:5];
        [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];

        MTLSize threadgroups    = MTLSizeMake((NSUInteger)total_q_heads, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake((NSUInteger)seq_len, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        copy_back_if_needed(bufOut, out, out_bytes);
        return 0;
    }
}

// ---------------------------------------------------------------------------
// GQA dispatch
// ---------------------------------------------------------------------------

int metal_gqa(const float* q, const float* k, const float* v, float* out,
              int batch, int num_heads, int num_kv_heads, int seq_len, int head_dim)
{
    @autoreleasepool {
        int total_q_heads  = batch * num_heads;
        int total_kv_heads = batch * num_kv_heads;

        NSUInteger q_bytes   = (NSUInteger)total_q_heads  * seq_len * head_dim * sizeof(float);
        NSUInteger kv_bytes  = (NSUInteger)total_kv_heads * seq_len * head_dim * sizeof(float);
        NSUInteger out_bytes = q_bytes;

        id<MTLBuffer> bufQ   = make_buf_ro(gAttnDevice, q, q_bytes);
        id<MTLBuffer> bufK   = make_buf_ro(gAttnDevice, k, kv_bytes);
        id<MTLBuffer> bufV   = make_buf_ro(gAttnDevice, v, kv_bytes);
        id<MTLBuffer> bufOut = make_buf_rw(gAttnDevice, out, out_bytes);
        if (!bufQ || !bufK || !bufV || !bufOut) return -1;

        NSUInteger smem_bytes = (NSUInteger)seq_len * seq_len * sizeof(float);

        id<MTLCommandBuffer>        cb  = [gAttnQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:gPSO_gqa];
        [enc setBuffer:bufQ   offset:0 atIndex:0];
        [enc setBuffer:bufK   offset:0 atIndex:1];
        [enc setBuffer:bufV   offset:0 atIndex:2];
        [enc setBuffer:bufOut offset:0 atIndex:3];
        [enc setBytes:&seq_len     length:sizeof(int) atIndex:4];
        [enc setBytes:&head_dim    length:sizeof(int) atIndex:5];
        [enc setBytes:&num_heads   length:sizeof(int) atIndex:6];
        [enc setBytes:&num_kv_heads length:sizeof(int) atIndex:7];
        [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];

        MTLSize threadgroups    = MTLSizeMake((NSUInteger)total_q_heads, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake((NSUInteger)seq_len, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        copy_back_if_needed(bufOut, out, out_bytes);
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Sliding Window dispatch
// ---------------------------------------------------------------------------

int metal_sliding_window(const float* q, const float* k, const float* v, float* out,
                         int batch, int num_heads, int seq_len, int head_dim, int window)
{
    @autoreleasepool {
        int total_heads = batch * num_heads;
        NSUInteger qkv_bytes = (NSUInteger)total_heads * seq_len * head_dim * sizeof(float);
        NSUInteger out_bytes = qkv_bytes;

        id<MTLBuffer> bufQ   = make_buf_ro(gAttnDevice, q, qkv_bytes);
        id<MTLBuffer> bufK   = make_buf_ro(gAttnDevice, k, qkv_bytes);
        id<MTLBuffer> bufV   = make_buf_ro(gAttnDevice, v, qkv_bytes);
        id<MTLBuffer> bufOut = make_buf_rw(gAttnDevice, out, out_bytes);
        if (!bufQ || !bufK || !bufV || !bufOut) return -1;

        NSUInteger smem_bytes = (NSUInteger)seq_len * seq_len * sizeof(float);

        id<MTLCommandBuffer>        cb  = [gAttnQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:gPSO_sw];
        [enc setBuffer:bufQ   offset:0 atIndex:0];
        [enc setBuffer:bufK   offset:0 atIndex:1];
        [enc setBuffer:bufV   offset:0 atIndex:2];
        [enc setBuffer:bufOut offset:0 atIndex:3];
        [enc setBytes:&seq_len  length:sizeof(int) atIndex:4];
        [enc setBytes:&head_dim length:sizeof(int) atIndex:5];
        [enc setBytes:&window   length:sizeof(int) atIndex:6];
        [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];

        MTLSize threadgroups    = MTLSizeMake((NSUInteger)total_heads, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake((NSUInteger)seq_len, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        copy_back_if_needed(bufOut, out, out_bytes);
        return 0;
    }
}
