#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "positional.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Static globals
// ---------------------------------------------------------------------------

static id<MTLDevice>               gPos_Device   = nil;
static id<MTLCommandQueue>         gPos_Queue    = nil;
static id<MTLComputePipelineState> gPSO_rope     = nil;
static id<MTLComputePipelineState> gPSO_alibi    = nil;

// ---------------------------------------------------------------------------

static id<MTLComputePipelineState> pos_make_pso(id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    return [gPos_Device newComputePipelineStateWithFunction:fn error:&err];
}

int metal_positional_init(const char* metallib_path) {
    @autoreleasepool {
        gPos_Device = MTLCreateSystemDefaultDevice();
        if (!gPos_Device) return -1;

        gPos_Queue = [gPos_Device newCommandQueue];
        if (!gPos_Queue) return -1;

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError* err = nil;
        id<MTLLibrary> lib = [gPos_Device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!lib) return -1;

        gPSO_rope  = pos_make_pso(lib, @"rope_kernel");
        gPSO_alibi = pos_make_pso(lib, @"alibi_kernel");

        if (!gPSO_rope || !gPSO_alibi) return -1;
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Generic buffer helper: creates MTLBuffer from host pointer (copy if needed)
// ---------------------------------------------------------------------------

static id<MTLBuffer> make_buf(id<MTLDevice> dev, const void* ptr, NSUInteger bytes) {
    vm_size_t page = getpagesize();
    if (((uintptr_t)ptr % page) == 0) {
        return [dev newBufferWithBytesNoCopy:(void*)ptr length:bytes
                                    options:MTLResourceStorageModeShared deallocator:nil];
    }
    return [dev newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> make_out_buf(id<MTLDevice> dev, void* ptr, NSUInteger bytes) {
    vm_size_t page = getpagesize();
    if (((uintptr_t)ptr % page) == 0) {
        return [dev newBufferWithBytesNoCopy:ptr length:bytes
                                    options:MTLResourceStorageModeShared deallocator:nil];
    }
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

// ---------------------------------------------------------------------------

int metal_rope(
    const float* x,
    float*       out,
    const float* cos_table,
    const float* sin_table,
    int          seq_len,
    int          head_dim,
    int          rope_mode,
    int          total_heads)
{
    @autoreleasepool {
        int num_pairs  = head_dim / 2;
        int grid_n     = total_heads * seq_len * num_pairs;
        NSUInteger xbytes   = (NSUInteger)(total_heads * seq_len * head_dim) * sizeof(float);
        NSUInteger tblbytes = (NSUInteger)(seq_len * num_pairs) * sizeof(float);

        id<MTLBuffer> bufX   = make_buf(gPos_Device, x,         xbytes);
        id<MTLBuffer> bufOut = make_out_buf(gPos_Device, out,   xbytes);
        id<MTLBuffer> bufCos = make_buf(gPos_Device, cos_table, tblbytes);
        id<MTLBuffer> bufSin = make_buf(gPos_Device, sin_table, tblbytes);
        if (!bufX || !bufOut || !bufCos || !bufSin) {
            [bufX release];
            [bufOut release];
            [bufCos release];
            [bufSin release];
            return -1;
        }

        id<MTLCommandBuffer>         cb  = [gPos_Queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_rope];
        [enc setBuffer:bufX   offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];
        [enc setBuffer:bufCos offset:0 atIndex:2];
        [enc setBuffer:bufSin offset:0 atIndex:3];
        [enc setBytes:&seq_len  length:sizeof(int) atIndex:4];
        [enc setBytes:&head_dim length:sizeof(int) atIndex:5];
        [enc setBytes:&rope_mode length:sizeof(int) atIndex:6];

        NSUInteger tw = gPSO_rope.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)grid_n, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        vm_size_t page = getpagesize();
        if (((uintptr_t)out % page) != 0) {
            memcpy(out, [bufOut contents], xbytes);
        }
        [bufX release];
        [bufOut release];
        [bufCos release];
        [bufSin release];
        return 0;
    }
}

int metal_rope_tensor(
    const void*  x,
    void*        out,
    const float* cos_table,
    const float* sin_table,
    int          seq_len,
    int          head_dim,
    int          rope_mode,
    int          total_heads)
{
    @autoreleasepool {
        if (!gPos_Queue || !gPSO_rope || !x || !out || !cos_table || !sin_table) return -1;
        if (seq_len <= 0 || head_dim <= 0 || total_heads <= 0 || (head_dim % 2) != 0) return -1;

        int num_pairs = head_dim / 2;
        int grid_n = total_heads * seq_len * num_pairs;
        NSUInteger tblbytes = (NSUInteger)(seq_len * num_pairs) * sizeof(float);

        id<MTLBuffer> bufX = (__bridge id)((void*)x);
        id<MTLBuffer> bufOut = (__bridge id)out;
        id<MTLBuffer> bufCos = make_buf(gPos_Device, cos_table, tblbytes);
        id<MTLBuffer> bufSin = make_buf(gPos_Device, sin_table, tblbytes);
        if (!bufX || !bufOut || !bufCos || !bufSin) {
            [bufCos release];
            [bufSin release];
            return -1;
        }

        id<MTLCommandBuffer> cb = [gPos_Queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) {
            [bufCos release];
            [bufSin release];
            return -1;
        }

        [enc setComputePipelineState:gPSO_rope];
        [enc setBuffer:bufX offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];
        [enc setBuffer:bufCos offset:0 atIndex:2];
        [enc setBuffer:bufSin offset:0 atIndex:3];
        [enc setBytes:&seq_len length:sizeof(int) atIndex:4];
        [enc setBytes:&head_dim length:sizeof(int) atIndex:5];
        [enc setBytes:&rope_mode length:sizeof(int) atIndex:6];

        NSUInteger tw = gPSO_rope.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)grid_n, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        int rc = cb.error ? -1 : 0;
        [bufCos release];
        [bufSin release];
        return rc;
    }
}

int metal_alibi(
    float*       out,
    const float* slopes,
    int          num_heads,
    int          seq_len_q,
    int          seq_len_k)
{
    @autoreleasepool {
        int grid_n = num_heads * seq_len_q * seq_len_k;
        NSUInteger outbytes    = (NSUInteger)grid_n * sizeof(float);
        NSUInteger slopebytes  = (NSUInteger)num_heads * sizeof(float);

        id<MTLBuffer> bufOut    = make_out_buf(gPos_Device, out, outbytes);
        id<MTLBuffer> bufSlopes = make_buf(gPos_Device, slopes, slopebytes);
        if (!bufOut || !bufSlopes) {
            [bufOut release];
            [bufSlopes release];
            return -1;
        }

        id<MTLCommandBuffer>         cb  = [gPos_Queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_alibi];
        [enc setBuffer:bufOut    offset:0 atIndex:0];
        [enc setBuffer:bufSlopes offset:0 atIndex:1];
        [enc setBytes:&seq_len_q length:sizeof(int) atIndex:2];
        [enc setBytes:&seq_len_k length:sizeof(int) atIndex:3];

        NSUInteger tw = gPSO_alibi.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)grid_n, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        vm_size_t page = getpagesize();
        if (((uintptr_t)out % page) != 0) {
            memcpy(out, [bufOut contents], outbytes);
        }
        [bufOut release];
        [bufSlopes release];
        return 0;
    }
}
