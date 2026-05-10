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
static id<MTLComputePipelineState> sPSO_viewHeads   = nil;
static id<MTLComputePipelineState> sPSO_mergeHeads  = nil;

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
        sPSO_viewHeads  = make_pso_s(lib, @"view_as_heads_kernel");
        sPSO_mergeHeads = make_pso_s(lib, @"merge_heads_kernel");

        if (!sPSO_transpose || !sPSO_copy || !sPSO_concat ||
            !sPSO_viewHeads || !sPSO_mergeHeads) return -1;

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
