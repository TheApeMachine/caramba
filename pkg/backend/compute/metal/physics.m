#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "physics.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Static globals — one Metal device, one command queue, one PSO per kernel.
// Initialized once via metal_physics_init.
// ---------------------------------------------------------------------------

static id<MTLDevice>               gPhys_Device   = nil;
static id<MTLCommandQueue>         gPhys_Queue    = nil;
static id<MTLComputePipelineState> gPSO_lap_1d    = nil;
static id<MTLComputePipelineState> gPSO_lap_2d    = nil;
static id<MTLComputePipelineState> gPSO_lap_3d    = nil;

// ---------------------------------------------------------------------------

static id<MTLComputePipelineState> phys_make_pso(id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    return [gPhys_Device newComputePipelineStateWithFunction:fn error:&err];
}

int metal_physics_init(const char* metallib_path) {
    @autoreleasepool {
        gPhys_Device = MTLCreateSystemDefaultDevice();
        if (!gPhys_Device) return -1;

        gPhys_Queue = [gPhys_Device newCommandQueue];
        if (!gPhys_Queue) return -1;

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError* err = nil;
        id<MTLLibrary> lib = [gPhys_Device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!lib) return -1;

        gPSO_lap_1d = phys_make_pso(lib, @"laplacian_1d_kernel");
        gPSO_lap_2d = phys_make_pso(lib, @"laplacian_2d_kernel");
        gPSO_lap_3d = phys_make_pso(lib, @"laplacian_3d_kernel");

        if (!gPSO_lap_1d || !gPSO_lap_2d || !gPSO_lap_3d) return -1;
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Buffer helpers — page-aligned host pointers get zero-copy bindings, others
// get device-side staging. Mirrors the convention used in positional.m.
// ---------------------------------------------------------------------------

static id<MTLBuffer> phys_make_buf(id<MTLDevice> dev, const void* ptr, NSUInteger bytes) {
    vm_size_t page = getpagesize();
    if (((uintptr_t)ptr % page) == 0) {
        return [dev newBufferWithBytesNoCopy:(void*)ptr length:bytes
                                    options:MTLResourceStorageModeShared deallocator:nil];
    }
    return [dev newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> phys_make_out_buf(id<MTLDevice> dev, void* ptr, NSUInteger bytes) {
    vm_size_t page = getpagesize();
    if (((uintptr_t)ptr % page) == 0) {
        return [dev newBufferWithBytesNoCopy:ptr length:bytes
                                    options:MTLResourceStorageModeShared deallocator:nil];
    }
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

// ---------------------------------------------------------------------------
// 1D Laplacian — host pointer entry point
// ---------------------------------------------------------------------------

int metal_laplacian_1d(const float* src, float* dst, int n, float inv_h2) {
    @autoreleasepool {
        if (!gPhys_Queue || !gPSO_lap_1d || !src || !dst || n <= 0) return -1;

        NSUInteger bytes = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bufSrc = phys_make_buf(gPhys_Device, src, bytes);
        id<MTLBuffer> bufDst = phys_make_out_buf(gPhys_Device, dst, bytes);
        if (!bufSrc || !bufDst) {
            [bufSrc release];
            [bufDst release];
            return -1;
        }

        id<MTLCommandBuffer>         cb  = [gPhys_Queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_lap_1d];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&n      length:sizeof(int)   atIndex:2];
        [enc setBytes:&inv_h2 length:sizeof(float) atIndex:3];

        NSUInteger tw = gPSO_lap_1d.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        vm_size_t page = getpagesize();
        if (((uintptr_t)dst % page) != 0) {
            memcpy(dst, [bufDst contents], bytes);
        }
        [bufSrc release];
        [bufDst release];
        return cb.error ? -1 : 0;
    }
}

int metal_laplacian_1d_tensor(const void* src_buffer, void* dst_buffer, int n, float inv_h2) {
    @autoreleasepool {
        if (!gPhys_Queue || !gPSO_lap_1d || !src_buffer || !dst_buffer || n <= 0) return -1;

        id<MTLBuffer> bufSrc = (__bridge id)((void*)src_buffer);
        id<MTLBuffer> bufDst = (__bridge id)dst_buffer;

        id<MTLCommandBuffer>         cb  = [gPhys_Queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) return -1;

        [enc setComputePipelineState:gPSO_lap_1d];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&n      length:sizeof(int)   atIndex:2];
        [enc setBytes:&inv_h2 length:sizeof(float) atIndex:3];

        NSUInteger tw = gPSO_lap_1d.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        return cb.error ? -1 : 0;
    }
}

// ---------------------------------------------------------------------------
// 2D Laplacian
// ---------------------------------------------------------------------------

int metal_laplacian_2d(const float* src, float* dst, int H, int W, float inv_h2) {
    @autoreleasepool {
        if (!gPhys_Queue || !gPSO_lap_2d || !src || !dst || H <= 0 || W <= 0) return -1;

        int total = H * W;
        NSUInteger bytes = (NSUInteger)total * sizeof(float);
        id<MTLBuffer> bufSrc = phys_make_buf(gPhys_Device, src, bytes);
        id<MTLBuffer> bufDst = phys_make_out_buf(gPhys_Device, dst, bytes);
        if (!bufSrc || !bufDst) {
            [bufSrc release];
            [bufDst release];
            return -1;
        }

        id<MTLCommandBuffer>         cb  = [gPhys_Queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_lap_2d];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&H      length:sizeof(int)   atIndex:2];
        [enc setBytes:&W      length:sizeof(int)   atIndex:3];
        [enc setBytes:&inv_h2 length:sizeof(float) atIndex:4];

        NSUInteger tw = gPSO_lap_2d.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        vm_size_t page = getpagesize();
        if (((uintptr_t)dst % page) != 0) {
            memcpy(dst, [bufDst contents], bytes);
        }
        [bufSrc release];
        [bufDst release];
        return cb.error ? -1 : 0;
    }
}

int metal_laplacian_2d_tensor(const void* src_buffer, void* dst_buffer, int H, int W, float inv_h2) {
    @autoreleasepool {
        if (!gPhys_Queue || !gPSO_lap_2d || !src_buffer || !dst_buffer || H <= 0 || W <= 0) return -1;

        int total = H * W;
        id<MTLBuffer> bufSrc = (__bridge id)((void*)src_buffer);
        id<MTLBuffer> bufDst = (__bridge id)dst_buffer;

        id<MTLCommandBuffer>         cb  = [gPhys_Queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) return -1;

        [enc setComputePipelineState:gPSO_lap_2d];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&H      length:sizeof(int)   atIndex:2];
        [enc setBytes:&W      length:sizeof(int)   atIndex:3];
        [enc setBytes:&inv_h2 length:sizeof(float) atIndex:4];

        NSUInteger tw = gPSO_lap_2d.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        return cb.error ? -1 : 0;
    }
}

// ---------------------------------------------------------------------------
// 3D Laplacian
// ---------------------------------------------------------------------------

int metal_laplacian_3d(const float* src, float* dst, int D, int H, int W, float inv_h2) {
    @autoreleasepool {
        if (!gPhys_Queue || !gPSO_lap_3d || !src || !dst || D <= 0 || H <= 0 || W <= 0) return -1;

        int total = D * H * W;
        NSUInteger bytes = (NSUInteger)total * sizeof(float);
        id<MTLBuffer> bufSrc = phys_make_buf(gPhys_Device, src, bytes);
        id<MTLBuffer> bufDst = phys_make_out_buf(gPhys_Device, dst, bytes);
        if (!bufSrc || !bufDst) {
            [bufSrc release];
            [bufDst release];
            return -1;
        }

        id<MTLCommandBuffer>         cb  = [gPhys_Queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_lap_3d];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&D      length:sizeof(int)   atIndex:2];
        [enc setBytes:&H      length:sizeof(int)   atIndex:3];
        [enc setBytes:&W      length:sizeof(int)   atIndex:4];
        [enc setBytes:&inv_h2 length:sizeof(float) atIndex:5];

        NSUInteger tw = gPSO_lap_3d.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        vm_size_t page = getpagesize();
        if (((uintptr_t)dst % page) != 0) {
            memcpy(dst, [bufDst contents], bytes);
        }
        [bufSrc release];
        [bufDst release];
        return cb.error ? -1 : 0;
    }
}

int metal_laplacian_3d_tensor(const void* src_buffer, void* dst_buffer, int D, int H, int W, float inv_h2) {
    @autoreleasepool {
        if (!gPhys_Queue || !gPSO_lap_3d || !src_buffer || !dst_buffer || D <= 0 || H <= 0 || W <= 0) return -1;

        int total = D * H * W;
        id<MTLBuffer> bufSrc = (__bridge id)((void*)src_buffer);
        id<MTLBuffer> bufDst = (__bridge id)dst_buffer;

        id<MTLCommandBuffer>         cb  = [gPhys_Queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) return -1;

        [enc setComputePipelineState:gPSO_lap_3d];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBytes:&D      length:sizeof(int)   atIndex:2];
        [enc setBytes:&H      length:sizeof(int)   atIndex:3];
        [enc setBytes:&W      length:sizeof(int)   atIndex:4];
        [enc setBytes:&inv_h2 length:sizeof(float) atIndex:5];

        NSUInteger tw = gPSO_lap_3d.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        return cb.error ? -1 : 0;
    }
}
