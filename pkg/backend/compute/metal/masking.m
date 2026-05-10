#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "masking.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Static globals — initialized once via metal_masking_init
// ---------------------------------------------------------------------------

static id<MTLDevice>               gMaskDevice    = nil;
static id<MTLCommandQueue>         gMaskQueue     = nil;
static id<MTLComputePipelineState> gPSO_causal    = nil;
static id<MTLComputePipelineState> gPSO_apply     = nil;

// ---------------------------------------------------------------------------

static id<MTLComputePipelineState> mask_make_pso(id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    return [gMaskDevice newComputePipelineStateWithFunction:fn error:&err];
}

int metal_masking_init(const char* metallib_path) {
    @autoreleasepool {
        gMaskDevice = MTLCreateSystemDefaultDevice();
        if (!gMaskDevice) return -1;

        gMaskQueue = [gMaskDevice newCommandQueue];
        if (!gMaskQueue) return -1;

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError* err   = nil;
        id<MTLLibrary> lib = [gMaskDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!lib) return -1;

        gPSO_causal = mask_make_pso(lib, @"causal_mask_kernel");
        gPSO_apply  = mask_make_pso(lib, @"apply_mask_kernel");

        if (!gPSO_causal || !gPSO_apply) return -1;
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static vm_size_t mask_page_size(void) { return (vm_size_t)getpagesize(); }

static id<MTLBuffer> make_buf_from(id<MTLDevice> dev, const void* ptr, NSUInteger bytes) {
    if (((uintptr_t)ptr % mask_page_size()) == 0) {
        return [dev newBufferWithBytesNoCopy:(void*)ptr
                                      length:bytes
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
    }
    return [dev newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> make_buf_out(id<MTLDevice> dev, void* ptr, NSUInteger bytes) {
    if (((uintptr_t)ptr % mask_page_size()) == 0) {
        return [dev newBufferWithBytesNoCopy:ptr
                                      length:bytes
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
    }
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

static void copy_back_if_needed(id<MTLBuffer> buf, void* dst, NSUInteger bytes) {
    if (((uintptr_t)dst % mask_page_size()) != 0) {
        memcpy(dst, [buf contents], bytes);
    }
}

// ---------------------------------------------------------------------------

int metal_causal_mask(float* out, int seq_len) {
    @autoreleasepool {
        NSUInteger out_bytes = (NSUInteger)(seq_len * seq_len) * sizeof(float);

        id<MTLBuffer> bufOut = make_buf_out(gMaskDevice, out, out_bytes);
        if (!bufOut) return -1;

        int sl = seq_len;
        id<MTLCommandBuffer>         cb  = [gMaskQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:gPSO_causal];
        [enc setBuffer:bufOut offset:0 atIndex:0];
        [enc setBytes:&sl length:sizeof(int) atIndex:1];

        // Dispatch 2D grid: (seq_len columns, seq_len rows)
        MTLSize threads     = MTLSizeMake((NSUInteger)seq_len, (NSUInteger)seq_len, 1);
        NSUInteger tw       = gPSO_causal.threadExecutionWidth;
        MTLSize threadgroup = MTLSizeMake(tw, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        copy_back_if_needed(bufOut, out, out_bytes);
        return 0;
    }
}

int metal_apply_mask(const float* scores, const float* mask, float* out, int n) {
    @autoreleasepool {
        NSUInteger bytes = (NSUInteger)n * sizeof(float);

        id<MTLBuffer> bufScores = make_buf_from(gMaskDevice, scores, bytes);
        id<MTLBuffer> bufMask   = make_buf_from(gMaskDevice, mask, bytes);
        id<MTLBuffer> bufOut    = make_buf_out(gMaskDevice, out, bytes);
        if (!bufScores || !bufMask || !bufOut) return -1;

        id<MTLCommandBuffer>         cb  = [gMaskQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:gPSO_apply];
        [enc setBuffer:bufScores offset:0 atIndex:0];
        [enc setBuffer:bufMask   offset:0 atIndex:1];
        [enc setBuffer:bufOut    offset:0 atIndex:2];

        NSUInteger tw       = gPSO_apply.threadExecutionWidth;
        MTLSize threads     = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize threadgroup = MTLSizeMake(tw, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        copy_back_if_needed(bufOut, out, bytes);
        return 0;
    }
}
