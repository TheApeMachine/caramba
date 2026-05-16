#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "shape.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Static globals — initialized once via metal_shape_init
// ---------------------------------------------------------------------------

static id<MTLDevice>               sDevice          = nil;
static id<MTLCommandQueue>         sQueue           = nil;
static id<MTLComputePipelineState> sPSO_transpose   = nil;
static id<MTLComputePipelineState> sPSO_copy        = nil;
static id<MTLComputePipelineState> sPSO_concat      = nil;
static id<MTLComputePipelineState> sPSO_split       = nil;
static id<MTLComputePipelineState> sPSO_upsample    = nil;
static id<MTLComputePipelineState> sPSO_viewHeads   = nil;
static id<MTLComputePipelineState> sPSO_mergeHeads  = nil;
static id<MTLComputePipelineState> sPSO_lastToken   = nil;

// ---------------------------------------------------------------------------

static id<MTLComputePipelineState> make_pso_s(id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    return [sDevice newComputePipelineStateWithFunction:fn error:&err];
}

int metal_shape_init(const char* metallib_path) {
    @autoreleasepool {
        sDevice = MTLCreateSystemDefaultDevice();
        if (!sDevice) return -1;

        sQueue = [sDevice newCommandQueue];
        if (!sQueue) return -1;

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError*  err  = nil;
        id<MTLLibrary> lib = [sDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!lib) return -1;

        sPSO_transpose  = make_pso_s(lib, @"transpose_kernel");
        sPSO_copy       = make_pso_s(lib, @"copy_kernel");
        sPSO_concat     = make_pso_s(lib, @"concat_kernel");
        sPSO_split      = make_pso_s(lib, @"split_kernel");
        sPSO_upsample   = make_pso_s(lib, @"upsample_nearest2d_kernel");
        sPSO_viewHeads  = make_pso_s(lib, @"view_as_heads_kernel");
        sPSO_mergeHeads = make_pso_s(lib, @"merge_heads_kernel");
        sPSO_lastToken  = make_pso_s(lib, @"last_token_kernel");

        if (!sPSO_transpose || !sPSO_copy || !sPSO_concat || !sPSO_split ||
            !sPSO_upsample || !sPSO_viewHeads || !sPSO_mergeHeads ||
            !sPSO_lastToken) return -1;

        return 0;
    }
}

// ---------------------------------------------------------------------------
// Helper: allocate an MTLBuffer wrapping host memory (page-aligned) or copy.
// ---------------------------------------------------------------------------

static id<MTLBuffer> make_buf(id<MTLDevice> dev, const void* ptr, NSUInteger bytes) {
    vm_size_t page = getpagesize();
    if (((uintptr_t)ptr % page) == 0) {
        return [dev newBufferWithBytesNoCopy:(void*)ptr
                                      length:bytes
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
    }
    return [dev newBufferWithBytes:ptr length:bytes
                           options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> make_dst_buf(id<MTLDevice> dev, void* ptr, NSUInteger bytes) {
    vm_size_t page = getpagesize();
    if (((uintptr_t)ptr % page) == 0) {
        return [dev newBufferWithBytesNoCopy:ptr
                                      length:bytes
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
    }
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

// Encode, commit, wait, and optionally copy back unaligned dst.
static void dispatch_commit(
    id<MTLComputePipelineState> pso,
    id<MTLCommandBuffer>* cb_out,
    NSArray<id<MTLBuffer>>* bufs,
    NSArray* scalars,      // alternating: (const void*, NSUInteger length) pairs — pass nil for none
    int grid_n,
    float* dst_host, id<MTLBuffer> dst_buf)
{
    @autoreleasepool {
        id<MTLCommandBuffer>        cb  = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:pso];

        for (NSUInteger i = 0; i < bufs.count; i++) {
            [enc setBuffer:bufs[i] offset:0 atIndex:i];
        }

        NSUInteger startIdx = bufs.count;
        if (scalars) {
            for (NSUInteger i = 0; i + 1 < scalars.count; i += 2) {
                NSData* bytes = scalars[i];
                [enc setBytes:bytes.bytes length:bytes.length atIndex:startIdx++];
            }
        }

        NSUInteger tw = pso.threadExecutionWidth;
        MTLSize threads    = MTLSizeMake((NSUInteger)grid_n, 1, 1);
        MTLSize threadgroup = MTLSizeMake(tw, 1, 1);

        [enc dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        // Copy back if dst_buf was not page-aligned.
        if (dst_host) {
            vm_size_t page = getpagesize();
            if (((uintptr_t)dst_host % page) != 0) {
                memcpy(dst_host, [dst_buf contents], [dst_buf length]);
            }
        }
    }
}

// ---------------------------------------------------------------------------

int metal_transpose(const float* src, float* dst,
                    const int* shape, int rank,
                    int dim0, int dim1, int n)
{
    @autoreleasepool {
        NSUInteger src_bytes   = (NSUInteger)n * sizeof(float);
        NSUInteger dst_bytes   = (NSUInteger)n * sizeof(float);
        NSUInteger shape_bytes = (NSUInteger)rank * sizeof(int);

        id<MTLBuffer> bufSrc   = make_buf(sDevice, src, src_bytes);
        id<MTLBuffer> bufDst   = make_dst_buf(sDevice, dst, dst_bytes);
        id<MTLBuffer> bufShape = make_buf(sDevice, shape, shape_bytes);

        if (!bufSrc || !bufDst || !bufShape) return -1;

        id<MTLCommandBuffer>        cb  = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sPSO_transpose];
        [enc setBuffer:bufSrc   offset:0 atIndex:0];
        [enc setBuffer:bufDst   offset:0 atIndex:1];
        [enc setBuffer:bufShape offset:0 atIndex:2];
        [enc setBytes:&rank  length:sizeof(int) atIndex:3];
        [enc setBytes:&dim0  length:sizeof(int) atIndex:4];
        [enc setBytes:&dim1  length:sizeof(int) atIndex:5];

        NSUInteger tw = sPSO_transpose.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        vm_size_t page = getpagesize();
        if (((uintptr_t)dst % page) != 0) {
            memcpy(dst, [bufDst contents], dst_bytes);
        }
        return 0;
    }
}

int metal_copy(const float* src, float* dst, int n) {
    @autoreleasepool {
        NSUInteger bytes = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bufSrc = make_buf(sDevice, src, bytes);
        id<MTLBuffer> bufDst = make_dst_buf(sDevice, dst, bytes);
        if (!bufSrc || !bufDst) return -1;

        id<MTLCommandBuffer>        cb  = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sPSO_copy];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        NSUInteger tw = sPSO_copy.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        vm_size_t page = getpagesize();
        if (((uintptr_t)dst % page) != 0) {
            memcpy(dst, [bufDst contents], bytes);
        }
        return 0;
    }
}

int metal_concat(const float* srcA, int n_a,
                 const float* srcB, int n_b,
                 float* dst)
{
    @autoreleasepool {
        int total = n_a + n_b;
        NSUInteger a_bytes = (NSUInteger)n_a * sizeof(float);
        NSUInteger b_bytes = (NSUInteger)n_b * sizeof(float);
        NSUInteger d_bytes = (NSUInteger)total * sizeof(float);

        id<MTLBuffer> bufA   = make_buf(sDevice, srcA, a_bytes);
        id<MTLBuffer> bufB   = make_buf(sDevice, srcB, b_bytes);
        id<MTLBuffer> bufDst = make_dst_buf(sDevice, dst, d_bytes);
        if (!bufA || !bufB || !bufDst) return -1;

        id<MTLCommandBuffer>        cb  = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sPSO_concat];
        [enc setBuffer:bufA   offset:0 atIndex:0];
        [enc setBuffer:bufB   offset:0 atIndex:1];
        [enc setBuffer:bufDst offset:0 atIndex:2];
        [enc setBytes:&n_a length:sizeof(int) atIndex:3];
        NSUInteger tw = sPSO_concat.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        vm_size_t page = getpagesize();
        if (((uintptr_t)dst % page) != 0) {
            memcpy(dst, [bufDst contents], d_bytes);
        }
        return 0;
    }
}

int metal_split(const float* src, float* dst,
                int outer, int dim_size, int split_size, int inner)
{
    @autoreleasepool {
        if (!src || !dst || outer <= 0 || dim_size <= 0 || split_size <= 0 || inner <= 0) {
            return -1;
        }

        if (dim_size % split_size != 0) return -1;

        int total = outer * dim_size * inner;
        NSUInteger bytes = (NSUInteger)total * sizeof(float);
        id<MTLBuffer> bufSrc = make_buf(sDevice, src, bytes);
        id<MTLBuffer> bufDst = make_dst_buf(sDevice, dst, bytes);
        if (!bufSrc || !bufDst || !sPSO_split) return -1;

        id<MTLCommandBuffer> cb = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sPSO_split];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&outer length:sizeof(int) atIndex:2];
        [enc setBytes:&dim_size length:sizeof(int) atIndex:3];
        [enc setBytes:&split_size length:sizeof(int) atIndex:4];
        [enc setBytes:&inner length:sizeof(int) atIndex:5];
        NSUInteger tw = sPSO_split.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        vm_size_t page = getpagesize();
        if (((uintptr_t)dst % page) != 0) {
            memcpy(dst, [bufDst contents], bytes);
        }
        return 0;
    }
}

int metal_upsample_nearest2d(const float* src, float* dst,
                             int B, int C, int H, int W,
                             int scale_h, int scale_w)
{
    @autoreleasepool {
        if (!src || !dst || B <= 0 || C <= 0 || H <= 0 || W <= 0 ||
            scale_h <= 0 || scale_w <= 0 || !sPSO_upsample) {
            return -1;
        }

        int input_n = B * C * H * W;
        int output_n = B * C * H * scale_h * W * scale_w;
        NSUInteger input_bytes = (NSUInteger)input_n * sizeof(float);
        NSUInteger output_bytes = (NSUInteger)output_n * sizeof(float);

        id<MTLBuffer> bufSrc = make_buf(sDevice, src, input_bytes);
        id<MTLBuffer> bufDst = make_dst_buf(sDevice, dst, output_bytes);
        if (!bufSrc || !bufDst) return -1;

        id<MTLCommandBuffer> cb = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sPSO_upsample];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&B length:sizeof(int) atIndex:2];
        [enc setBytes:&C length:sizeof(int) atIndex:3];
        [enc setBytes:&H length:sizeof(int) atIndex:4];
        [enc setBytes:&W length:sizeof(int) atIndex:5];
        [enc setBytes:&scale_h length:sizeof(int) atIndex:6];
        [enc setBytes:&scale_w length:sizeof(int) atIndex:7];
        NSUInteger tw = sPSO_upsample.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)output_n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        vm_size_t page = getpagesize();
        if (((uintptr_t)dst % page) != 0) {
            memcpy(dst, [bufDst contents], output_bytes);
        }
        return 0;
    }
}

int metal_view_as_heads(const float* src, float* dst,
                        int B, int T, int H, int head_dim)
{
    @autoreleasepool {
        int n = B * T * H * head_dim;
        NSUInteger bytes = (NSUInteger)n * sizeof(float);

        id<MTLBuffer> bufSrc = make_buf(sDevice, src, bytes);
        id<MTLBuffer> bufDst = make_dst_buf(sDevice, dst, bytes);
        if (!bufSrc || !bufDst) return -1;

        id<MTLCommandBuffer>        cb  = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sPSO_viewHeads];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&B        length:sizeof(int) atIndex:2];
        [enc setBytes:&T        length:sizeof(int) atIndex:3];
        [enc setBytes:&H        length:sizeof(int) atIndex:4];
        [enc setBytes:&head_dim length:sizeof(int) atIndex:5];
        NSUInteger tw = sPSO_viewHeads.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        vm_size_t page = getpagesize();
        if (((uintptr_t)dst % page) != 0) {
            memcpy(dst, [bufDst contents], bytes);
        }
        return 0;
    }
}

int metal_merge_heads(const float* src, float* dst,
                      int B, int H, int T, int head_dim)
{
    @autoreleasepool {
        int n = B * H * T * head_dim;
        NSUInteger bytes = (NSUInteger)n * sizeof(float);

        id<MTLBuffer> bufSrc = make_buf(sDevice, src, bytes);
        id<MTLBuffer> bufDst = make_dst_buf(sDevice, dst, bytes);
        if (!bufSrc || !bufDst) return -1;

        id<MTLCommandBuffer>        cb  = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sPSO_mergeHeads];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&B        length:sizeof(int) atIndex:2];
        [enc setBytes:&H        length:sizeof(int) atIndex:3];
        [enc setBytes:&T        length:sizeof(int) atIndex:4];
        [enc setBytes:&head_dim length:sizeof(int) atIndex:5];
        NSUInteger tw = sPSO_mergeHeads.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        vm_size_t page = getpagesize();
        if (((uintptr_t)dst % page) != 0) {
            memcpy(dst, [bufDst contents], bytes);
        }
        return 0;
    }
}

int metal_last_token(const float* src, float* dst,
                     int outer, int seq_len, int feature)
{
    @autoreleasepool {
        if (!src || !dst || outer <= 0 || seq_len <= 0 || feature <= 0) {
            return -1;
        }

        int input_n = outer * seq_len * feature;
        int output_n = outer * feature;
        NSUInteger input_bytes = (NSUInteger)input_n * sizeof(float);
        NSUInteger output_bytes = (NSUInteger)output_n * sizeof(float);

        id<MTLBuffer> bufSrc = make_buf(sDevice, src, input_bytes);
        id<MTLBuffer> bufDst = make_dst_buf(sDevice, dst, output_bytes);
        if (!bufSrc || !bufDst || !sPSO_lastToken) return -1;

        id<MTLCommandBuffer> cb = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sPSO_lastToken];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&seq_len length:sizeof(int) atIndex:2];
        [enc setBytes:&feature length:sizeof(int) atIndex:3];
        NSUInteger tw = sPSO_lastToken.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)output_n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        vm_size_t page = getpagesize();
        if (((uintptr_t)dst % page) != 0) {
            memcpy(dst, [bufDst contents], output_bytes);
        }
        return 0;
    }
}

static int dispatch_shape_tensor(
	id<MTLComputePipelineState> pso,
	const void* src,
    void* dst,
    int a,
    int b,
    int c,
    int d,
    int n)
{
    @autoreleasepool {
        if (!sQueue || !pso || !src || !dst || n <= 0) return -1;

        id<MTLBuffer> bufSrc = (__bridge id)((void*)src);
        id<MTLBuffer> bufDst = (__bridge id)dst;
        id<MTLCommandBuffer> cb = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&a length:sizeof(int) atIndex:2];
        [enc setBytes:&b length:sizeof(int) atIndex:3];
        [enc setBytes:&c length:sizeof(int) atIndex:4];
        [enc setBytes:&d length:sizeof(int) atIndex:5];
        NSUInteger tw = pso.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return 0;
    }
}

int metal_view_as_heads_tensor(const void* src, void* dst,
                               int B, int T, int H, int head_dim)
{
    return dispatch_shape_tensor(sPSO_viewHeads, src, dst, B, T, H, head_dim,
                                 B * T * H * head_dim);
}

int metal_merge_heads_tensor(const void* src, void* dst,
                             int B, int H, int T, int head_dim)
{
    return dispatch_shape_tensor(sPSO_mergeHeads, src, dst, B, H, T, head_dim,
                                 B * H * T * head_dim);
}

int metal_last_token_tensor(const void* src, void* dst,
                            int outer, int seq_len, int feature)
{
    @autoreleasepool {
        int n = outer * feature;
        if (!sQueue || !sPSO_lastToken || !src || !dst || n <= 0 ||
            seq_len <= 0 || feature <= 0) return -1;

        id<MTLBuffer> bufSrc = (__bridge id)((void*)src);
        id<MTLBuffer> bufDst = (__bridge id)dst;
        id<MTLCommandBuffer> cb = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sPSO_lastToken];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&seq_len length:sizeof(int) atIndex:2];
        [enc setBytes:&feature length:sizeof(int) atIndex:3];
        NSUInteger tw = sPSO_lastToken.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return 0;
    }
}

int metal_copy_tensor(const void* src, void* dst, int n)
{
    @autoreleasepool {
        if (!sQueue || !sPSO_copy || !src || !dst || n <= 0) return -1;

        id<MTLBuffer> bufSrc = (__bridge id)((void*)src);
        id<MTLBuffer> bufDst = (__bridge id)dst;
        id<MTLCommandBuffer> cb = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sPSO_copy];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        NSUInteger tw = sPSO_copy.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
}

int metal_concat_tensor(const void* srcA, int n_a,
                        const void* srcB, int n_b,
                        void* dst)
{
    @autoreleasepool {
        int n = n_a + n_b;
        if (!sQueue || !sPSO_concat || !dst || n <= 0 ||
            n_a < 0 || n_b < 0 || (n_a > 0 && !srcA) || (n_b > 0 && !srcB)) {
            return -1;
        }

        id<MTLBuffer> bufA = (__bridge id)((void*)srcA);
        id<MTLBuffer> bufB = (__bridge id)((void*)srcB);
        id<MTLBuffer> bufDst = (__bridge id)dst;
        id<MTLCommandBuffer> cb = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sPSO_concat];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufDst offset:0 atIndex:2];
        [enc setBytes:&n_a length:sizeof(int) atIndex:3];
        NSUInteger tw = sPSO_concat.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
}

int metal_split_tensor(const void* src, void* dst,
                       int outer, int dim_size, int split_size, int inner)
{
    @autoreleasepool {
        if (!sQueue || !sPSO_split || !src || !dst || outer <= 0 ||
            dim_size <= 0 || split_size <= 0 || inner <= 0 ||
            split_size > dim_size || dim_size % split_size != 0) {
            return -1;
        }

        int n = outer * dim_size * inner;
        if (n <= 0) return -1;

        id<MTLBuffer> bufSrc = (__bridge id)((void*)src);
        id<MTLBuffer> bufDst = (__bridge id)dst;
        id<MTLCommandBuffer> cb = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sPSO_split];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&outer length:sizeof(int) atIndex:2];
        [enc setBytes:&dim_size length:sizeof(int) atIndex:3];
        [enc setBytes:&split_size length:sizeof(int) atIndex:4];
        [enc setBytes:&inner length:sizeof(int) atIndex:5];
        NSUInteger tw = sPSO_split.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
}

int metal_transpose_tensor(const void* src, void* dst,
                           const int* shape, int rank,
                           int dim0, int dim1, int n)
{
    @autoreleasepool {
        if (!sQueue || !sPSO_transpose || !src || !dst || !shape ||
            rank <= 0 || rank > 8 || dim0 < 0 || dim1 < 0 ||
            dim0 >= rank || dim1 >= rank || n <= 0) return -1;

        id<MTLBuffer> bufSrc = (__bridge id)((void*)src);
        id<MTLBuffer> bufDst = (__bridge id)dst;
        id<MTLCommandBuffer> cb = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sPSO_transpose];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:shape length:(NSUInteger)rank * sizeof(int) atIndex:2];
        [enc setBytes:&rank length:sizeof(int) atIndex:3];
        [enc setBytes:&dim0 length:sizeof(int) atIndex:4];
        [enc setBytes:&dim1 length:sizeof(int) atIndex:5];
        NSUInteger tw = sPSO_transpose.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
}

int metal_upsample_nearest2d_tensor(const void* src, void* dst,
                                    int B, int C, int H, int W,
                                    int scale_h, int scale_w)
{
    @autoreleasepool {
        int n = B * C * H * scale_h * W * scale_w;
        if (!sQueue || !sPSO_upsample || !src || !dst || B <= 0 || C <= 0 ||
            H <= 0 || W <= 0 || scale_h <= 0 || scale_w <= 0 || n <= 0) {
            return -1;
        }

        id<MTLBuffer> bufSrc = (__bridge id)((void*)src);
        id<MTLBuffer> bufDst = (__bridge id)dst;
        id<MTLCommandBuffer> cb = [sQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:sPSO_upsample];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&B length:sizeof(int) atIndex:2];
        [enc setBytes:&C length:sizeof(int) atIndex:3];
        [enc setBytes:&H length:sizeof(int) atIndex:4];
        [enc setBytes:&W length:sizeof(int) atIndex:5];
        [enc setBytes:&scale_h length:sizeof(int) atIndex:6];
        [enc setBytes:&scale_w length:sizeof(int) atIndex:7];
        NSUInteger tw = sPSO_upsample.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return 0;
    }
}
