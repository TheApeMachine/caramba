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
static id<MTLComputePipelineState> gPSO_kv_append = nil;
static id<MTLComputePipelineState> gPSO_kv_repack = nil;
static id<MTLComputePipelineState> gPSO_kv_write = nil;

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
        gPSO_kv_append = attn_make_pso(lib, @"kv_append_forward");
        gPSO_kv_repack = attn_make_pso(lib, @"kv_repack_forward");
        gPSO_kv_write = attn_make_pso(lib, @"kv_write_forward");

        if (!gPSO_sdpa || !gPSO_mqa || !gPSO_gqa || !gPSO_sw ||
            !gPSO_kv_append || !gPSO_kv_repack || !gPSO_kv_write) return -1;

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
               int batch, int num_heads, int query_len, int key_value_len,
               int head_dim, int causal)
{
    @autoreleasepool {
        if (batch <= 0 || num_heads <= 0 || query_len <= 0 ||
            key_value_len <= 0 || head_dim <= 0) return -1;
        int total_heads = batch * num_heads;
        int key_value_stride = key_value_len;
        NSUInteger q_bytes = (NSUInteger)total_heads * query_len * head_dim * sizeof(float);
        NSUInteger kv_bytes = (NSUInteger)total_heads * key_value_len * head_dim * sizeof(float);
        NSUInteger out_bytes = q_bytes;

        id<MTLBuffer> bufQ   = make_buf_ro(gAttnDevice, q, q_bytes);
        id<MTLBuffer> bufK   = make_buf_ro(gAttnDevice, k, kv_bytes);
        id<MTLBuffer> bufV   = make_buf_ro(gAttnDevice, v, kv_bytes);
        id<MTLBuffer> bufOut = make_buf_rw(gAttnDevice, out, out_bytes);
        if (!bufQ || !bufK || !bufV || !bufOut) return -1;

        // Threadgroup shared memory: query_len * key_value_len floats.
        NSUInteger smem_bytes = (NSUInteger)query_len * key_value_len * sizeof(float);

        id<MTLCommandBuffer>        cb  = [gAttnQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:gPSO_sdpa];
        [enc setBuffer:bufQ   offset:0 atIndex:0];
        [enc setBuffer:bufK   offset:0 atIndex:1];
        [enc setBuffer:bufV   offset:0 atIndex:2];
        [enc setBuffer:bufOut offset:0 atIndex:3];
        [enc setBytes:&query_len     length:sizeof(int) atIndex:4];
        [enc setBytes:&key_value_len length:sizeof(int) atIndex:5];
        [enc setBytes:&key_value_stride length:sizeof(int) atIndex:6];
        [enc setBytes:&head_dim      length:sizeof(int) atIndex:7];
        [enc setBytes:&causal        length:sizeof(int) atIndex:8];
        [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];

        MTLSize threadgroups    = MTLSizeMake((NSUInteger)total_heads, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake((NSUInteger)query_len, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        copy_back_if_needed(bufOut, out, out_bytes);
        return 0;
    }
}

int metal_sdpa_tensor(const void* q, const void* k, const void* v, void* out,
                      int batch, int num_heads, int query_len, int key_value_len,
                      int key_value_stride, int head_dim, int causal)
{
    @autoreleasepool {
        if (!gAttnQueue || !gPSO_sdpa || !q || !k || !v || !out) return -1;
        if (batch <= 0 || num_heads <= 0 || query_len <= 0 ||
            key_value_len <= 0 || key_value_stride < key_value_len ||
            head_dim <= 0) return -1;

        int total_heads = batch * num_heads;
        NSUInteger smem_bytes = (NSUInteger)query_len * key_value_len * sizeof(float);

        id<MTLBuffer> bufQ   = (__bridge id)((void*)q);
        id<MTLBuffer> bufK   = (__bridge id)((void*)k);
        id<MTLBuffer> bufV   = (__bridge id)((void*)v);
        id<MTLBuffer> bufOut = (__bridge id)out;

        id<MTLCommandBuffer>        cb  = [gAttnQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:gPSO_sdpa];
        [enc setBuffer:bufQ   offset:0 atIndex:0];
        [enc setBuffer:bufK   offset:0 atIndex:1];
        [enc setBuffer:bufV   offset:0 atIndex:2];
        [enc setBuffer:bufOut offset:0 atIndex:3];
        [enc setBytes:&query_len     length:sizeof(int) atIndex:4];
        [enc setBytes:&key_value_len length:sizeof(int) atIndex:5];
        [enc setBytes:&key_value_stride length:sizeof(int) atIndex:6];
        [enc setBytes:&head_dim      length:sizeof(int) atIndex:7];
        [enc setBytes:&causal        length:sizeof(int) atIndex:8];
        [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];

        MTLSize threadgroups    = MTLSizeMake((NSUInteger)total_heads, 1, 1);
        MTLSize threadsPerGroup = MTLSizeMake((NSUInteger)query_len, 1, 1);
        [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        return 0;
    }
}

int metal_kv_append_tensor(const void* previous_key, const void* previous_value,
                           const void* key_chunk, const void* value_chunk,
                           void* output_key, void* output_value,
                           int batch, int num_heads, int previous_len,
                           int chunk_len, int head_dim)
{
    @autoreleasepool {
        if (!gAttnQueue || !gPSO_kv_append || !key_chunk || !value_chunk ||
            !output_key || !output_value) return -1;
        if (batch <= 0 || num_heads <= 0 || previous_len < 0 ||
            chunk_len <= 0 || head_dim <= 0) return -1;

        int total_len = previous_len + chunk_len;
        int total = batch * num_heads * total_len * head_dim;
        if (total <= 0) return -1;

        id<MTLBuffer> previousKey = previous_key
            ? (__bridge id)((void*)previous_key)
            : (__bridge id)((void*)key_chunk);
        id<MTLBuffer> previousValue = previous_value
            ? (__bridge id)((void*)previous_value)
            : (__bridge id)((void*)value_chunk);
        id<MTLBuffer> chunkKey    = (__bridge id)((void*)key_chunk);
        id<MTLBuffer> chunkValue  = (__bridge id)((void*)value_chunk);
        id<MTLBuffer> outKey      = (__bridge id)output_key;
        id<MTLBuffer> outValue    = (__bridge id)output_value;

        id<MTLCommandBuffer>         cb  = [gAttnQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:gPSO_kv_append];
        [enc setBuffer:previousKey   offset:0 atIndex:0];
        [enc setBuffer:previousValue offset:0 atIndex:1];
        [enc setBuffer:chunkKey      offset:0 atIndex:2];
        [enc setBuffer:chunkValue    offset:0 atIndex:3];
        [enc setBuffer:outKey        offset:0 atIndex:4];
        [enc setBuffer:outValue      offset:0 atIndex:5];
        [enc setBytes:&previous_len length:sizeof(int) atIndex:6];
        [enc setBytes:&chunk_len    length:sizeof(int) atIndex:7];
        [enc setBytes:&head_dim     length:sizeof(int) atIndex:8];
        [enc setBytes:&total_len    length:sizeof(int) atIndex:9];

        NSUInteger tw = gPSO_kv_append.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        return 0;
    }
}

int metal_kv_repack_tensor(const void* previous_key, const void* previous_value,
                           void* output_key, void* output_value,
                           int batch, int num_heads, int current_len,
                           int head_dim, int previous_capacity,
                           int output_capacity)
{
    @autoreleasepool {
        if (!gAttnQueue || !gPSO_kv_repack || !previous_key || !previous_value ||
            !output_key || !output_value) return -1;
        if (batch <= 0 || num_heads <= 0 || current_len <= 0 || head_dim <= 0 ||
            previous_capacity < current_len || output_capacity < current_len) return -1;

        int total = batch * num_heads * current_len * head_dim;
        if (total <= 0) return -1;

        id<MTLBuffer> previousKey   = (__bridge id)((void*)previous_key);
        id<MTLBuffer> previousValue = (__bridge id)((void*)previous_value);
        id<MTLBuffer> outKey        = (__bridge id)output_key;
        id<MTLBuffer> outValue      = (__bridge id)output_value;

        id<MTLCommandBuffer>         cb  = [gAttnQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:gPSO_kv_repack];
        [enc setBuffer:previousKey   offset:0 atIndex:0];
        [enc setBuffer:previousValue offset:0 atIndex:1];
        [enc setBuffer:outKey        offset:0 atIndex:2];
        [enc setBuffer:outValue      offset:0 atIndex:3];
        [enc setBytes:&current_len       length:sizeof(int) atIndex:4];
        [enc setBytes:&head_dim          length:sizeof(int) atIndex:5];
        [enc setBytes:&previous_capacity length:sizeof(int) atIndex:6];
        [enc setBytes:&output_capacity   length:sizeof(int) atIndex:7];

        NSUInteger tw = gPSO_kv_repack.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        return 0;
    }
}

int metal_kv_write_tensor(void* cache_key, void* cache_value,
                          const void* key_chunk, const void* value_chunk,
                          int batch, int num_heads, int start_len,
                          int chunk_len, int head_dim, int capacity)
{
    @autoreleasepool {
        if (!gAttnQueue || !gPSO_kv_write || !cache_key || !cache_value ||
            !key_chunk || !value_chunk) return -1;
        if (batch <= 0 || num_heads <= 0 || start_len < 0 || chunk_len <= 0 ||
            head_dim <= 0 || capacity < start_len + chunk_len) return -1;

        int total = batch * num_heads * chunk_len * head_dim;
        if (total <= 0) return -1;

        id<MTLBuffer> cacheKey   = (__bridge id)cache_key;
        id<MTLBuffer> cacheValue = (__bridge id)cache_value;
        id<MTLBuffer> chunkKey   = (__bridge id)((void*)key_chunk);
        id<MTLBuffer> chunkValue = (__bridge id)((void*)value_chunk);

        id<MTLCommandBuffer>         cb  = [gAttnQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:gPSO_kv_write];
        [enc setBuffer:cacheKey   offset:0 atIndex:0];
        [enc setBuffer:cacheValue offset:0 atIndex:1];
        [enc setBuffer:chunkKey   offset:0 atIndex:2];
        [enc setBuffer:chunkValue offset:0 atIndex:3];
        [enc setBytes:&start_len length:sizeof(int) atIndex:4];
        [enc setBytes:&chunk_len length:sizeof(int) atIndex:5];
        [enc setBytes:&head_dim  length:sizeof(int) atIndex:6];
        [enc setBytes:&capacity  length:sizeof(int) atIndex:7];

        NSUInteger tw = gPSO_kv_write.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

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
        [enc setBytes:&seq_len   length:sizeof(int) atIndex:4];
        [enc setBytes:&head_dim  length:sizeof(int) atIndex:5];
        [enc setBytes:&num_heads length:sizeof(int) atIndex:6];
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

static int dispatch_gqa_buffers(id<MTLBuffer> bufQ, id<MTLBuffer> bufK,
                                id<MTLBuffer> bufV, id<MTLBuffer> bufOut,
                                int batch, int num_heads, int num_kv_heads,
                                int query_len, int key_value_len,
                                int key_value_stride, int head_dim,
                                int causal)
{
    if (!gAttnQueue || !gPSO_gqa || !bufQ || !bufK || !bufV || !bufOut) return -1;
    if (batch <= 0 || num_heads <= 0 || num_kv_heads <= 0 ||
        query_len <= 0 || key_value_len < query_len ||
        key_value_stride < key_value_len || head_dim <= 0 ||
        num_heads % num_kv_heads != 0) return -1;

    NSUInteger smem_bytes = (NSUInteger)query_len * key_value_len * sizeof(float);
    int total_q_heads = batch * num_heads;

    id<MTLCommandBuffer>        cb  = [gAttnQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!cb || !enc) return -1;

    [enc setComputePipelineState:gPSO_gqa];
    [enc setBuffer:bufQ   offset:0 atIndex:0];
    [enc setBuffer:bufK   offset:0 atIndex:1];
    [enc setBuffer:bufV   offset:0 atIndex:2];
    [enc setBuffer:bufOut offset:0 atIndex:3];
    [enc setBytes:&query_len        length:sizeof(int) atIndex:4];
    [enc setBytes:&key_value_len    length:sizeof(int) atIndex:5];
    [enc setBytes:&key_value_stride length:sizeof(int) atIndex:6];
    [enc setBytes:&head_dim         length:sizeof(int) atIndex:7];
    [enc setBytes:&num_heads        length:sizeof(int) atIndex:8];
    [enc setBytes:&num_kv_heads     length:sizeof(int) atIndex:9];
    [enc setBytes:&causal           length:sizeof(int) atIndex:10];
    [enc setThreadgroupMemoryLength:smem_bytes atIndex:0];

    MTLSize threadgroups    = MTLSizeMake((NSUInteger)total_q_heads, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake((NSUInteger)query_len, 1, 1);
    [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    return cb.error ? -1 : 0;
}

int metal_gqa(const float* q, const float* k, const float* v, float* out,
              int batch, int num_heads, int num_kv_heads, int seq_len, int head_dim,
              int causal)
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

        int rc = dispatch_gqa_buffers(bufQ, bufK, bufV, bufOut, batch, num_heads,
                                      num_kv_heads, seq_len, seq_len, seq_len,
                                      head_dim, causal);
        if (rc != 0) return rc;

        copy_back_if_needed(bufOut, out, out_bytes);
        return 0;
    }
}

int metal_gqa_tensor(const void* q, const void* k, const void* v, void* out,
                     int batch, int num_heads, int num_kv_heads,
                     int query_len, int key_value_len, int key_value_stride,
                     int head_dim, int causal)
{
    @autoreleasepool {
        id<MTLBuffer> bufQ   = (__bridge id)((void*)q);
        id<MTLBuffer> bufK   = (__bridge id)((void*)k);
        id<MTLBuffer> bufV   = (__bridge id)((void*)v);
        id<MTLBuffer> bufOut = (__bridge id)out;

        return dispatch_gqa_buffers(bufQ, bufK, bufV, bufOut, batch, num_heads,
                                    num_kv_heads, query_len, key_value_len,
                                    key_value_stride, head_dim, causal);
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
