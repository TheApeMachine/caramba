#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "vsa.h"
#include <math.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Static globals — initialized once via metal_vsa_init
// ---------------------------------------------------------------------------

static id<MTLDevice>               gVDevice  = nil;
static id<MTLCommandQueue>         gVQueue   = nil;
static id<MTLComputePipelineState> gPSO_bind = nil;
static id<MTLComputePipelineState> gPSO_mul  = nil;
static id<MTLComputePipelineState> gPSO_scale = nil;

static id<MTLComputePipelineState> make_vpso(id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    return [gVDevice newComputePipelineStateWithFunction:fn error:&err];
}

int metal_vsa_init(const char* metallib_path) {
    @autoreleasepool {
        gVDevice = MTLCreateSystemDefaultDevice();
        if (!gVDevice) return -1;

        gVQueue = [gVDevice newCommandQueue];
        if (!gVQueue) return -1;

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError* err = nil;
        id<MTLLibrary> lib = [gVDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!lib) return -1;

        gPSO_bind  = make_vpso(lib, @"vsa_bind_kernel");
        gPSO_mul   = make_vpso(lib, @"vsa_mul_kernel");
        gPSO_scale = make_vpso(lib, @"vsa_scale_kernel");

        if (!gPSO_bind || !gPSO_mul || !gPSO_scale) return -1;

        return 0;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static id<MTLBuffer> vsa_buf_ro(const void* ptr, NSUInteger bytes) {
    return [gVDevice newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> vsa_buf_rw(NSUInteger bytes) {
    return [gVDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

static void vsa_commit_wait(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
}

// ---------------------------------------------------------------------------
// metal_vsa_bind
// ---------------------------------------------------------------------------

int metal_vsa_bind(const float* a, const float* b, float* out, int n) {
    @autoreleasepool {
        if (!gVQueue || !gPSO_bind) return -1;

        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bufA   = vsa_buf_ro(a, nb);
        id<MTLBuffer> bufB   = vsa_buf_ro(b, nb);
        id<MTLBuffer> bufOut = vsa_buf_rw(nb);
        id<MTLBuffer> bufN   = [gVDevice newBufferWithBytes:&n length:sizeof(int)
                                                    options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [gVQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_bind];
        [enc setBuffer:bufA   offset:0 atIndex:0];
        [enc setBuffer:bufB   offset:0 atIndex:1];
        [enc setBuffer:bufOut offset:0 atIndex:2];
        [enc setBuffer:bufN   offset:0 atIndex:3];

        NSUInteger tw = gPSO_bind.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
         threadsPerThreadgroup:MTLSizeMake(tw,1,1)];
        [enc endEncoding];
        vsa_commit_wait(cb);

        memcpy(out, [bufOut contents], nb);
        return 0;
    }
}

// ---------------------------------------------------------------------------
// metal_vsa_l2normalize
// ---------------------------------------------------------------------------

int metal_vsa_l2normalize(const float* in, float* out, int n) {
    @autoreleasepool {
        if (!gVQueue || !gPSO_scale) return -1;

        // Compute norm on host (input is already in host memory)
        double sumsq = 0.0;
        for (int i = 0; i < n; i++) sumsq += (double)in[i] * (double)in[i];
        double norm = sqrt(sumsq);
        float inv_norm = (norm > 1e-12) ? (float)(1.0 / norm) : 1.0f;

        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bufIn    = vsa_buf_ro(in, nb);
        id<MTLBuffer> bufOut   = vsa_buf_rw(nb);
        id<MTLBuffer> bufScale = [gVDevice newBufferWithBytes:&inv_norm length:sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufN     = [gVDevice newBufferWithBytes:&n length:sizeof(int)
                                                      options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [gVQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_scale];
        [enc setBuffer:bufIn    offset:0 atIndex:0];
        [enc setBuffer:bufOut   offset:0 atIndex:1];
        [enc setBuffer:bufScale offset:0 atIndex:2];
        [enc setBuffer:bufN     offset:0 atIndex:3];

        NSUInteger tw = gPSO_scale.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
         threadsPerThreadgroup:MTLSizeMake(tw,1,1)];
        [enc endEncoding];
        vsa_commit_wait(cb);

        memcpy(out, [bufOut contents], nb);
        return 0;
    }
}

// ---------------------------------------------------------------------------
// metal_vsa_dot
// ---------------------------------------------------------------------------

int metal_vsa_dot(const float* a, const float* b, float* out, int n) {
    @autoreleasepool {
        if (!gVQueue || !gPSO_mul) return -1;

        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bufA   = vsa_buf_ro(a, nb);
        id<MTLBuffer> bufB   = vsa_buf_ro(b, nb);
        id<MTLBuffer> bufProd = vsa_buf_rw(nb);
        id<MTLBuffer> bufN   = [gVDevice newBufferWithBytes:&n length:sizeof(int)
                                                    options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [gVQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_mul];
        [enc setBuffer:bufA    offset:0 atIndex:0];
        [enc setBuffer:bufB    offset:0 atIndex:1];
        [enc setBuffer:bufProd offset:0 atIndex:2];
        [enc setBuffer:bufN    offset:0 atIndex:3];

        NSUInteger tw = gPSO_mul.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
         threadsPerThreadgroup:MTLSizeMake(tw,1,1)];
        [enc endEncoding];
        vsa_commit_wait(cb);

        // Reduce on host
        const float* prod = (const float*)[bufProd contents];
        double sum = 0.0;
        for (int i = 0; i < n; i++) sum += (double)prod[i];
        out[0] = (float)sum;

        return 0;
    }
}
