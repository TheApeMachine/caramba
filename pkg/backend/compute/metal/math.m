#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "math.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Static globals — initialized once via metal_math_init
// ---------------------------------------------------------------------------

static id<MTLDevice>               gMDevice        = nil;
static id<MTLCommandQueue>         gMQueue         = nil;
static id<MTLComputePipelineState> gPSO_matmul     = nil;
static id<MTLComputePipelineState> gPSO_add        = nil;
static id<MTLComputePipelineState> gPSO_mul        = nil;
static id<MTLComputePipelineState> gPSO_isdscale   = nil;
static id<MTLComputePipelineState> gPSO_exp        = nil;
static id<MTLComputePipelineState> gPSO_log        = nil;
static id<MTLComputePipelineState> gPSO_softmax    = nil;
static id<MTLComputePipelineState> gPSO_layernorm  = nil;
static id<MTLComputePipelineState> gPSO_rmsnorm    = nil;

static id<MTLComputePipelineState> make_mpso(id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    return [gMDevice newComputePipelineStateWithFunction:fn error:&err];
}

int metal_math_init(const char* metallib_path) {
    @autoreleasepool {
        gMDevice = MTLCreateSystemDefaultDevice();
        if (!gMDevice) return -1;

        gMQueue = [gMDevice newCommandQueue];
        if (!gMQueue) return -1;

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError* err = nil;
        id<MTLLibrary> lib = [gMDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!lib) return -1;

        gPSO_matmul    = make_mpso(lib, @"matmul_kernel");
        gPSO_add       = make_mpso(lib, @"add_kernel");
        gPSO_mul       = make_mpso(lib, @"mul_kernel");
        gPSO_isdscale  = make_mpso(lib, @"inv_sqrt_dim_scale_kernel");
        gPSO_exp       = make_mpso(lib, @"exp_kernel");
        gPSO_log       = make_mpso(lib, @"log_kernel");
        gPSO_softmax   = make_mpso(lib, @"softmax_kernel");
        gPSO_layernorm = make_mpso(lib, @"layernorm_kernel");
        gPSO_rmsnorm   = make_mpso(lib, @"rmsnorm_kernel");

        if (!gPSO_matmul || !gPSO_add || !gPSO_mul || !gPSO_isdscale ||
            !gPSO_exp    || !gPSO_log || !gPSO_softmax ||
            !gPSO_layernorm || !gPSO_rmsnorm) return -1;

        return 0;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static id<MTLBuffer> make_buf_ro(id<MTLDevice> dev, const void* ptr, NSUInteger bytes) {
    vm_size_t page = getpagesize();
    if (((uintptr_t)ptr % page) == 0 && bytes > 0) {
        return [dev newBufferWithBytesNoCopy:(void*)ptr length:bytes
                                    options:MTLResourceStorageModeShared deallocator:nil];
    }
    return [dev newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> make_buf_rw(id<MTLDevice> dev, void* ptr, NSUInteger bytes) {
    vm_size_t page = getpagesize();
    if (((uintptr_t)ptr % page) == 0 && bytes > 0) {
        return [dev newBufferWithBytesNoCopy:ptr length:bytes
                                     options:MTLResourceStorageModeShared deallocator:nil];
    }
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

static void copy_back_if_needed(id<MTLBuffer> buf, void* dst, NSUInteger bytes) {
    vm_size_t page = getpagesize();
    if (((uintptr_t)dst % page) != 0) {
        memcpy(dst, [buf contents], bytes);
    }
}

static id<MTLCommandBuffer> begin_cb(void) {
    return [gMQueue commandBuffer];
}

static void commit_wait(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
}

// ---------------------------------------------------------------------------

int metal_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    @autoreleasepool {
        NSUInteger ab = (NSUInteger)(M*K)*sizeof(float);
        NSUInteger bb = (NSUInteger)(K*N)*sizeof(float);
        NSUInteger cb = (NSUInteger)(M*N)*sizeof(float);

        id<MTLBuffer> bufA = make_buf_ro(gMDevice, A, ab);
        id<MTLBuffer> bufB = make_buf_ro(gMDevice, B, bb);
        id<MTLBuffer> bufC = make_buf_rw(gMDevice, C, cb);
        if (!bufA || !bufB || !bufC) return -1;

        unsigned int dims[3] = { (unsigned int)M, (unsigned int)K, (unsigned int)N };
        id<MTLBuffer> bufDims = [gMDevice newBufferWithBytes:dims length:sizeof(dims)
                                                     options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:gPSO_matmul];
        [enc setBuffer:bufA   offset:0 atIndex:0];
        [enc setBuffer:bufB   offset:0 atIndex:1];
        [enc setBuffer:bufC   offset:0 atIndex:2];
        [enc setBuffer:bufDims offset:0 atIndex:3];

        NSUInteger ts = 16;
        MTLSize tg   = MTLSizeMake(ts, ts, 1);
        MTLSize grid = MTLSizeMake(((NSUInteger)N+ts-1)/ts * ts,
                                   ((NSUInteger)M+ts-1)/ts * ts, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        commit_wait(cmdBuf);
        copy_back_if_needed(bufC, C, cb);
        return 0;
    }
}

int metal_add(const float* a, const float* b, float* out, int n) {
    @autoreleasepool {
        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bA   = make_buf_ro(gMDevice, a, nb);
        id<MTLBuffer> bB   = make_buf_ro(gMDevice, b, nb);
        id<MTLBuffer> bOut = make_buf_rw(gMDevice, out, nb);
        if (!bA || !bB || !bOut) return -1;

        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_add];
        [enc setBuffer:bA   offset:0 atIndex:0];
        [enc setBuffer:bB   offset:0 atIndex:1];
        [enc setBuffer:bOut offset:0 atIndex:2];
        NSUInteger tw = gPSO_add.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(tw,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bOut, out, nb);
        return 0;
    }
}

int metal_mul(const float* a, const float* b, float* out, int n) {
    @autoreleasepool {
        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bA   = make_buf_ro(gMDevice, a, nb);
        id<MTLBuffer> bB   = make_buf_ro(gMDevice, b, nb);
        id<MTLBuffer> bOut = make_buf_rw(gMDevice, out, nb);
        if (!bA || !bB || !bOut) return -1;

        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_mul];
        [enc setBuffer:bA   offset:0 atIndex:0];
        [enc setBuffer:bB   offset:0 atIndex:1];
        [enc setBuffer:bOut offset:0 atIndex:2];
        NSUInteger tw = gPSO_mul.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(tw,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bOut, out, nb);
        return 0;
    }
}

int metal_inv_sqrt_dim_scale(const float* src, float* dst, int n, int dim) {
    @autoreleasepool {
        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bSrc = make_buf_ro(gMDevice, src, nb);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, nb);
        if (!bSrc || !bDst) return -1;

        float scale = 1.0f / __builtin_sqrtf((float)dim);
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_isdscale];
        [enc setBuffer:bSrc offset:0 atIndex:0];
        [enc setBuffer:bDst offset:0 atIndex:1];
        [enc setBytes:&scale length:sizeof(float) atIndex:2];
        NSUInteger tw = gPSO_isdscale.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(tw,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, nb);
        return 0;
    }
}

int metal_exp(const float* src, float* dst, int n) {
    @autoreleasepool {
        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bSrc = make_buf_ro(gMDevice, src, nb);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, nb);
        if (!bSrc || !bDst) return -1;
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_exp];
        [enc setBuffer:bSrc offset:0 atIndex:0];
        [enc setBuffer:bDst offset:0 atIndex:1];
        NSUInteger tw = gPSO_exp.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(tw,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, nb);
        return 0;
    }
}

int metal_log(const float* src, float* dst, int n) {
    @autoreleasepool {
        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bSrc = make_buf_ro(gMDevice, src, nb);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, nb);
        if (!bSrc || !bDst) return -1;
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_log];
        [enc setBuffer:bSrc offset:0 atIndex:0];
        [enc setBuffer:bDst offset:0 atIndex:1];
        NSUInteger tw = gPSO_log.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(tw,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, nb);
        return 0;
    }
}

int metal_softmax(const float* src, float* dst, int num_rows, int dim_size) {
    @autoreleasepool {
        NSUInteger nb = (NSUInteger)(num_rows * dim_size) * sizeof(float);
        id<MTLBuffer> bSrc = make_buf_ro(gMDevice, src, nb);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, nb);
        if (!bSrc || !bDst) return -1;
        unsigned int ds = (unsigned int)dim_size;
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_softmax];
        [enc setBuffer:bSrc offset:0 atIndex:0];
        [enc setBuffer:bDst offset:0 atIndex:1];
        [enc setBytes:&ds length:sizeof(unsigned int) atIndex:2];
        NSUInteger tgs = (NSUInteger)dim_size < 256 ? (NSUInteger)dim_size : 256;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_rows,1,1)
        threadsPerThreadgroup:MTLSizeMake(tgs,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, nb);
        return 0;
    }
}

int metal_layernorm(const float* src, float* dst,
                    const float* weight, const float* bias,
                    int num_rows, int d_model, float eps) {
    @autoreleasepool {
        NSUInteger nb  = (NSUInteger)(num_rows * d_model) * sizeof(float);
        NSUInteger nb1 = (NSUInteger)d_model * sizeof(float);
        id<MTLBuffer> bSrc = make_buf_ro(gMDevice, src, nb);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, nb);
        id<MTLBuffer> bW   = make_buf_ro(gMDevice, weight, nb1);
        id<MTLBuffer> bB   = make_buf_ro(gMDevice, bias,   nb1);
        if (!bSrc || !bDst || !bW || !bB) return -1;
        unsigned int dm = (unsigned int)d_model;
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_layernorm];
        [enc setBuffer:bSrc offset:0 atIndex:0];
        [enc setBuffer:bDst offset:0 atIndex:1];
        [enc setBuffer:bW   offset:0 atIndex:2];
        [enc setBuffer:bB   offset:0 atIndex:3];
        [enc setBytes:&dm  length:sizeof(unsigned int) atIndex:4];
        [enc setBytes:&eps length:sizeof(float) atIndex:5];
        NSUInteger tgs = (NSUInteger)d_model < 256 ? (NSUInteger)d_model : 256;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_rows,1,1)
        threadsPerThreadgroup:MTLSizeMake(tgs,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, nb);
        return 0;
    }
}

int metal_rmsnorm(const float* src, float* dst,
                  const float* weight,
                  int num_rows, int d_model, float eps) {
    @autoreleasepool {
        NSUInteger nb  = (NSUInteger)(num_rows * d_model) * sizeof(float);
        NSUInteger nb1 = (NSUInteger)d_model * sizeof(float);
        id<MTLBuffer> bSrc = make_buf_ro(gMDevice, src, nb);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, nb);
        id<MTLBuffer> bW   = make_buf_ro(gMDevice, weight, nb1);
        if (!bSrc || !bDst || !bW) return -1;
        unsigned int dm = (unsigned int)d_model;
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_rmsnorm];
        [enc setBuffer:bSrc offset:0 atIndex:0];
        [enc setBuffer:bDst offset:0 atIndex:1];
        [enc setBuffer:bW   offset:0 atIndex:2];
        [enc setBytes:&dm  length:sizeof(unsigned int) atIndex:3];
        [enc setBytes:&eps length:sizeof(float) atIndex:4];
        NSUInteger tgs = (NSUInteger)d_model < 256 ? (NSUInteger)d_model : 256;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_rows,1,1)
        threadsPerThreadgroup:MTLSizeMake(tgs,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, nb);
        return 0;
    }
}
