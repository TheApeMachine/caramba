#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_kernel_math.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Static globals — initialized once via metal_math_init
// ---------------------------------------------------------------------------

static id<MTLDevice>               gMDevice        = nil;
static id<MTLCommandQueue>         gMQueue         = nil;
static id<MTLComputePipelineState> gPSO_matmul     = nil;
static id<MTLComputePipelineState> gPSO_matmul_add = nil;
static id<MTLComputePipelineState> gPSO_matmul_add_gelu = nil;
static id<MTLComputePipelineState> gPSO_add        = nil;
static id<MTLComputePipelineState> gPSO_mul        = nil;
static id<MTLComputePipelineState> gPSO_isdscale   = nil;
static id<MTLComputePipelineState> gPSO_exp        = nil;
static id<MTLComputePipelineState> gPSO_log        = nil;
static id<MTLComputePipelineState> gPSO_softmax    = nil;
static id<MTLComputePipelineState> gPSO_logsumexp  = nil;
static id<MTLComputePipelineState> gPSO_dropout    = nil;
static id<MTLComputePipelineState> gPSO_layernorm  = nil;
static id<MTLComputePipelineState> gPSO_rmsnorm    = nil;
static id<MTLComputePipelineState> gPSO_sign        = nil;
static id<MTLComputePipelineState> gPSO_outer       = nil;
static id<MTLComputePipelineState> gPSO_axpy        = nil;
static id<MTLComputePipelineState> gPSO_scale2      = nil;
static id<MTLComputePipelineState> gPSO_sqrt_vec    = nil;
static id<MTLComputePipelineState> gPSO_add_scalar  = nil;
static id<MTLComputePipelineState> gPSO_div_vec     = nil;
static id<MTLComputePipelineState> gPSO_clamp_vec   = nil;
static id<MTLComputePipelineState> gPSO_mse_loss    = nil;
static id<MTLComputePipelineState> gPSO_mse_grad    = nil;
static id<MTLComputePipelineState> gPSO_ce_stats    = nil;
static id<MTLComputePipelineState> gPSO_ce_loss     = nil;
static id<MTLComputePipelineState> gPSO_ce_grad     = nil;
static id<MTLComputePipelineState> gPSO_accuracy    = nil;
static id<MTLComputePipelineState> gPSO_f1_counts   = nil;

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
        gPSO_matmul_add = make_mpso(lib, @"matmul_add_kernel");
        gPSO_matmul_add_gelu = make_mpso(lib, @"matmul_add_gelu_kernel");
        gPSO_add       = make_mpso(lib, @"add_kernel");
        gPSO_mul       = make_mpso(lib, @"mul_kernel");
        gPSO_isdscale  = make_mpso(lib, @"inv_sqrt_dim_scale_kernel");
        gPSO_exp       = make_mpso(lib, @"exp_kernel");
        gPSO_log       = make_mpso(lib, @"log_kernel");
        gPSO_softmax   = make_mpso(lib, @"softmax_kernel");
        gPSO_logsumexp = make_mpso(lib, @"logsumexp_kernel");
        gPSO_dropout   = make_mpso(lib, @"dropout_kernel");
        gPSO_layernorm = make_mpso(lib, @"layernorm_kernel");
        gPSO_rmsnorm   = make_mpso(lib, @"rmsnorm_kernel");
        gPSO_sign       = make_mpso(lib, @"sign_kernel");
        gPSO_outer      = make_mpso(lib, @"outer_kernel");
        gPSO_axpy       = make_mpso(lib, @"axpy_kernel");
        gPSO_scale2     = make_mpso(lib, @"scale_kernel2");
        gPSO_sqrt_vec   = make_mpso(lib, @"sqrt_vec_kernel");
        gPSO_add_scalar = make_mpso(lib, @"add_scalar_kernel");
        gPSO_div_vec    = make_mpso(lib, @"div_vec_kernel");
        gPSO_clamp_vec  = make_mpso(lib, @"clamp_vec_kernel");
        gPSO_mse_loss   = make_mpso(lib, @"train_mse_loss_kernel");
        gPSO_mse_grad   = make_mpso(lib, @"train_mse_grad_kernel");
        gPSO_ce_stats   = make_mpso(lib, @"train_ce_stats_kernel");
        gPSO_ce_loss    = make_mpso(lib, @"train_cross_entropy_loss_kernel");
        gPSO_ce_grad    = make_mpso(lib, @"train_cross_entropy_grad_kernel");
        gPSO_accuracy   = make_mpso(lib, @"bench_accuracy_kernel");
        gPSO_f1_counts  = make_mpso(lib, @"bench_f1_counts_kernel");

        if (!gPSO_matmul || !gPSO_matmul_add || !gPSO_matmul_add_gelu ||
            !gPSO_add || !gPSO_mul || !gPSO_isdscale ||
            !gPSO_exp || !gPSO_log || !gPSO_softmax ||
            !gPSO_logsumexp || !gPSO_dropout ||
            !gPSO_layernorm || !gPSO_rmsnorm ||
            !gPSO_sign || !gPSO_outer ||
            !gPSO_axpy || !gPSO_scale2 || !gPSO_sqrt_vec ||
            !gPSO_add_scalar || !gPSO_div_vec || !gPSO_clamp_vec ||
            !gPSO_mse_loss || !gPSO_mse_grad || !gPSO_ce_stats ||
            !gPSO_ce_loss || !gPSO_ce_grad || !gPSO_accuracy ||
            !gPSO_f1_counts) return -1;

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

static int dispatch_binary_tensor(
    id<MTLComputePipelineState> pso,
    const void* a,
    const void* b,
    void* out,
    int n)
{
    @autoreleasepool {
        if (!gMQueue || !pso || !a || !b || !out) return -1;

        id<MTLBuffer> bufA = (__bridge id)((void*)a);
        id<MTLBuffer> bufB = (__bridge id)((void*)b);
        id<MTLBuffer> bufOut = (__bridge id)out;

        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufOut offset:0 atIndex:2];
        NSUInteger tw = pso.threadExecutionWidth;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(tw,1,1)];
        [enc endEncoding];
        commit_wait(cb);

        return 0;
    }
}

// Shared tiled matmul encode/submit for resident buffers (also used after host uploads).
static int dispatch_matmul_buffers(
    id<MTLBuffer> bufA,
    id<MTLBuffer> bufB,
    id<MTLBuffer> bufC,
    int M, int K, int N)
{
    if (!gMQueue || !gPSO_matmul || !bufA || !bufB || !bufC) return -1;

    unsigned int dims[3] = { (unsigned int)M, (unsigned int)K, (unsigned int)N };
    id<MTLBuffer> bufDims = [gMDevice newBufferWithBytes:dims length:sizeof(dims)
                                                 options:MTLResourceStorageModeShared];
    if (!bufDims) return -1;

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

    return 0;
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

        int rc = dispatch_matmul_buffers(bufA, bufB, bufC, M, K, N);

        if (rc != 0) return rc;

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

int metal_matmul_tensor(const void* A, const void* B, void* C, int M, int K, int N) {
    @autoreleasepool {
        if (!A || !B || !C) return -1;

        id<MTLBuffer> bufA = (__bridge id)((void*)A);
        id<MTLBuffer> bufB = (__bridge id)((void*)B);
        id<MTLBuffer> bufC = (__bridge id)C;

        return dispatch_matmul_buffers(bufA, bufB, bufC, M, K, N);
    }
}

int metal_add_tensor(const void* a, const void* b, void* out, int n) {
    return dispatch_binary_tensor(gPSO_add, a, b, out, n);
}

int metal_mul_tensor(const void* a, const void* b, void* out, int n) {
    return dispatch_binary_tensor(gPSO_mul, a, b, out, n);
}

int metal_matmul_add_tensor(
    const void* A, const void* B, const void* bias, void* C,
    int M, int K, int N, int bias_n, int gelu)
{
    @autoreleasepool {
        id<MTLComputePipelineState> pso = gelu ? gPSO_matmul_add_gelu : gPSO_matmul_add;
        if (!gMQueue || !pso || !A || !B || !bias || !C) return -1;
        if (M <= 0 || K <= 0 || N <= 0) return -1;
        if (bias_n <= 0) return -1;
        if (bias_n != N && bias_n != M * N) return -1;
        if (gelu != 0 && gelu != 1) return -1;

        id<MTLBuffer> bufA = (__bridge id)((void*)A);
        id<MTLBuffer> bufB = (__bridge id)((void*)B);
        id<MTLBuffer> bufBias = (__bridge id)((void*)bias);
        id<MTLBuffer> bufC = (__bridge id)C;

        unsigned int dims[4] = {
            (unsigned int)M,
            (unsigned int)K,
            (unsigned int)N,
            (unsigned int)bias_n
        };
        id<MTLBuffer> bufDims = [gMDevice newBufferWithBytes:dims length:sizeof(dims)
                                                     options:MTLResourceStorageModeShared];
        if (!bufDims) return -1;

        id<MTLCommandBuffer> cmdBuf = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufBias offset:0 atIndex:2];
        [enc setBuffer:bufC offset:0 atIndex:3];
        [enc setBuffer:bufDims offset:0 atIndex:4];

        NSUInteger ts = 16;
        MTLSize tg = MTLSizeMake(ts, ts, 1);
        MTLSize grid = MTLSizeMake(((NSUInteger)N+ts-1)/ts * ts,
                                   ((NSUInteger)M+ts-1)/ts * ts, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        commit_wait(cmdBuf);

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

int metal_logsumexp(const float* src, float* dst, int num_rows, int dim_size) {
    @autoreleasepool {
        if (!src || !dst || num_rows <= 0 || dim_size <= 0) return -1;

        NSUInteger srcBytes = (NSUInteger)(num_rows * dim_size) * sizeof(float);
        NSUInteger dstBytes = (NSUInteger)num_rows * sizeof(float);
        id<MTLBuffer> bSrc = make_buf_ro(gMDevice, src, srcBytes);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, dstBytes);
        if (!bSrc || !bDst || !gPSO_logsumexp) return -1;

        unsigned int ds = (unsigned int)dim_size;
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_logsumexp];
        [enc setBuffer:bSrc offset:0 atIndex:0];
        [enc setBuffer:bDst offset:0 atIndex:1];
        [enc setBytes:&ds length:sizeof(unsigned int) atIndex:2];
        NSUInteger tgs = (NSUInteger)dim_size < 256 ? (NSUInteger)dim_size : 256;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_rows,1,1)
        threadsPerThreadgroup:MTLSizeMake(tgs,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, dstBytes);
        return 0;
    }
}

int metal_dropout(const float* src, float* dst, int n, float p, int training, int seed) {
    @autoreleasepool {
        if (!src || !dst || n < 0 || p < 0.0f || p >= 1.0f) return -1;
        if (n == 0) return 0;

        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bSrc = make_buf_ro(gMDevice, src, nb);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, nb);
        if (!bSrc || !bDst || !gPSO_dropout) return -1;

        unsigned int trainingValue = training ? 1u : 0u;
        unsigned int seedValue = (unsigned int)seed;
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_dropout];
        [enc setBuffer:bSrc offset:0 atIndex:0];
        [enc setBuffer:bDst offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(float) atIndex:2];
        [enc setBytes:&trainingValue length:sizeof(unsigned int) atIndex:3];
        [enc setBytes:&seedValue length:sizeof(unsigned int) atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(gPSO_dropout.threadExecutionWidth,1,1)];
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

int metal_sign(const float* src, float* dst, int n) {
    @autoreleasepool {
        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bSrc = make_buf_ro(gMDevice, src, nb);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, nb);
        if (!bSrc || !bDst) return -1;
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_sign];
        [enc setBuffer:bSrc offset:0 atIndex:0];
        [enc setBuffer:bDst offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, nb);
        return 0;
    }
}

int metal_outer(const float* a, const float* b, float* dst, int M, int N) {
    @autoreleasepool {
        NSUInteger ab  = (NSUInteger)M * sizeof(float);
        NSUInteger bb  = (NSUInteger)N * sizeof(float);
        NSUInteger db  = (NSUInteger)(M * N) * sizeof(float);
        id<MTLBuffer> bA   = make_buf_ro(gMDevice, a,   ab);
        id<MTLBuffer> bB   = make_buf_ro(gMDevice, b,   bb);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, db);
        if (!bA || !bB || !bDst) return -1;
        unsigned int dims[2] = { (unsigned int)M, (unsigned int)N };
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_outer];
        [enc setBuffer:bA   offset:0 atIndex:0];
        [enc setBuffer:bB   offset:0 atIndex:1];
        [enc setBuffer:bDst offset:0 atIndex:2];
        [enc setBytes:dims  length:sizeof(dims) atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)N,(NSUInteger)M,1)
        threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, db);
        return 0;
    }
}

int metal_axpy(float* dst, const float* src, float scale, int n) {
    @autoreleasepool {
        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, nb);
        id<MTLBuffer> bSrc = make_buf_ro(gMDevice, src, nb);
        if (!bDst || !bSrc) return -1;
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_axpy];
        [enc setBuffer:bDst offset:0 atIndex:0];
        [enc setBuffer:bSrc offset:0 atIndex:1];
        [enc setBytes:&scale length:sizeof(float) atIndex:2];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, nb);
        return 0;
    }
}

int metal_scale(float* dst, float s, int n) {
    @autoreleasepool {
        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, nb);
        if (!bDst) return -1;
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_scale2];
        [enc setBuffer:bDst offset:0 atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, nb);
        return 0;
    }
}

int metal_sqrt_vec(const float* src, float* dst, int n) {
    @autoreleasepool {
        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bSrc = make_buf_ro(gMDevice, src, nb);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, nb);
        if (!bSrc || !bDst) return -1;
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_sqrt_vec];
        [enc setBuffer:bSrc offset:0 atIndex:0];
        [enc setBuffer:bDst offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, nb);
        return 0;
    }
}

int metal_add_scalar(float* dst, float scalar, int n) {
    @autoreleasepool {
        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, nb);
        if (!bDst) return -1;
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_add_scalar];
        [enc setBuffer:bDst offset:0 atIndex:0];
        [enc setBytes:&scalar length:sizeof(float) atIndex:1];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, nb);
        return 0;
    }
}

int metal_div_vec(const float* a, const float* b, float* dst, int n) {
    @autoreleasepool {
        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bA   = make_buf_ro(gMDevice, a, nb);
        id<MTLBuffer> bB   = make_buf_ro(gMDevice, b, nb);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, nb);
        if (!bA || !bB || !bDst) return -1;
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_div_vec];
        [enc setBuffer:bA   offset:0 atIndex:0];
        [enc setBuffer:bB   offset:0 atIndex:1];
        [enc setBuffer:bDst offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, nb);
        return 0;
    }
}

int metal_clamp_vec(float* dst, float lo, float hi, int n) {
    @autoreleasepool {
        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bDst = make_buf_rw(gMDevice, dst, nb);
        if (!bDst) return -1;
        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_clamp_vec];
        [enc setBuffer:bDst offset:0 atIndex:0];
        [enc setBytes:&lo length:sizeof(float) atIndex:1];
        [enc setBytes:&hi length:sizeof(float) atIndex:2];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bDst, dst, nb);
        return 0;
    }
}

int metal_train_mse_loss(const float* predictions, const float* targets, float* out, int n) {
    @autoreleasepool {
        if (!predictions || !targets || !out || n <= 0 || !gPSO_mse_loss) return -1;

        NSUInteger nb = (NSUInteger)n * sizeof(float);
        float zero = 0.0f;
        unsigned int un = (unsigned int)n;
        id<MTLBuffer> bPred = make_buf_ro(gMDevice, predictions, nb);
        id<MTLBuffer> bTgt = make_buf_ro(gMDevice, targets, nb);
        id<MTLBuffer> bOut = [gMDevice newBufferWithBytes:&zero length:sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        if (!bPred || !bTgt || !bOut) return -1;

        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_mse_loss];
        [enc setBuffer:bPred offset:0 atIndex:0];
        [enc setBuffer:bTgt offset:0 atIndex:1];
        [enc setBuffer:bOut offset:0 atIndex:2];
        [enc setBytes:&un length:sizeof(unsigned int) atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(gPSO_mse_loss.threadExecutionWidth,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        memcpy(out, [bOut contents], sizeof(float));
        return 0;
    }
}

int metal_train_mse_grad(const float* predictions, const float* targets, float* out, int n) {
    @autoreleasepool {
        if (!predictions || !targets || !out || n < 0 || !gPSO_mse_grad) return -1;
        if (n == 0) return 0;

        NSUInteger nb = (NSUInteger)n * sizeof(float);
        unsigned int un = (unsigned int)n;
        id<MTLBuffer> bPred = make_buf_ro(gMDevice, predictions, nb);
        id<MTLBuffer> bTgt = make_buf_ro(gMDevice, targets, nb);
        id<MTLBuffer> bOut = make_buf_rw(gMDevice, out, nb);
        if (!bPred || !bTgt || !bOut) return -1;

        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_mse_grad];
        [enc setBuffer:bPred offset:0 atIndex:0];
        [enc setBuffer:bTgt offset:0 atIndex:1];
        [enc setBuffer:bOut offset:0 atIndex:2];
        [enc setBytes:&un length:sizeof(unsigned int) atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(gPSO_mse_grad.threadExecutionWidth,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        copy_back_if_needed(bOut, out, nb);
        return 0;
    }
}

static int metal_cross_entropy_common(
    const float* logits, const float* targets, float* out, int n, int gradient)
{
    @autoreleasepool {
        if (!logits || !targets || !out || n <= 0 || !gPSO_ce_stats) return -1;
        if (gradient && !gPSO_ce_grad) return -1;
        if (!gradient && !gPSO_ce_loss) return -1;

        NSUInteger nb = (NSUInteger)n * sizeof(float);
        unsigned int un = (unsigned int)n;
        float zero = 0.0f;
        float statsZero[2] = {0.0f, 0.0f};
        id<MTLBuffer> bLogits = make_buf_ro(gMDevice, logits, nb);
        id<MTLBuffer> bTargets = make_buf_ro(gMDevice, targets, nb);
        id<MTLBuffer> bStats = [gMDevice newBufferWithBytes:statsZero length:sizeof(statsZero)
                                                     options:MTLResourceStorageModeShared];
        if (!bLogits || !bTargets || !bStats) return -1;

        id<MTLBuffer> bOut = nil;
        if (gradient) {
            bOut = make_buf_rw(gMDevice, out, nb);
        } else {
            bOut = [gMDevice newBufferWithBytes:&zero length:sizeof(float)
                                        options:MTLResourceStorageModeShared];
        }
        if (!bOut) return -1;

        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_ce_stats];
        [enc setBuffer:bLogits offset:0 atIndex:0];
        [enc setBuffer:bStats offset:0 atIndex:1];
        [enc setBytes:&un length:sizeof(unsigned int) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [enc endEncoding];

        id<MTLComputePipelineState> cePSO = gradient ? gPSO_ce_grad : gPSO_ce_loss;
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:cePSO];
        [enc setBuffer:bLogits offset:0 atIndex:0];
        [enc setBuffer:bTargets offset:0 atIndex:1];
        [enc setBuffer:bOut offset:0 atIndex:2];
        [enc setBuffer:bStats offset:0 atIndex:3];
        [enc setBytes:&un length:sizeof(unsigned int) atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(cePSO.threadExecutionWidth,1,1)];
        [enc endEncoding];
        commit_wait(cb);

        if (gradient) {
            copy_back_if_needed(bOut, out, nb);
        } else {
            memcpy(out, [bOut contents], sizeof(float));
        }
        return 0;
    }
}

int metal_train_cross_entropy_loss(const float* logits, const float* targets, float* out, int n) {
    return metal_cross_entropy_common(logits, targets, out, n, 0);
}

int metal_train_cross_entropy_grad(const float* logits, const float* targets, float* out, int n) {
    return metal_cross_entropy_common(logits, targets, out, n, 1);
}

int metal_bench_accuracy(const float* predictions, const float* targets, float* out, int n) {
    @autoreleasepool {
        if (!predictions || !targets || !out || n <= 0 || !gPSO_accuracy) return -1;

        NSUInteger nb = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> bPred = make_buf_ro(gMDevice, predictions, nb);
        id<MTLBuffer> bTgt = make_buf_ro(gMDevice, targets, nb);
        id<MTLBuffer> bOut = [gMDevice newBufferWithLength:sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        unsigned int un = (unsigned int)n;
        if (!bPred || !bTgt || !bOut) return -1;

        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_accuracy];
        [enc setBuffer:bPred offset:0 atIndex:0];
        [enc setBuffer:bTgt offset:0 atIndex:1];
        [enc setBuffer:bOut offset:0 atIndex:2];
        [enc setBytes:&un length:sizeof(unsigned int) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1)
        threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        memcpy(out, [bOut contents], sizeof(float));
        return 0;
    }
}

int metal_bench_f1_counts(const float* predictions, const float* targets, float* out, int n) {
    @autoreleasepool {
        if (!predictions || !targets || !out || n < 0 || !gPSO_f1_counts) return -1;
        if (n == 0) {
            out[0] = 0.0f;
            out[1] = 0.0f;
            out[2] = 0.0f;
            return 0;
        }

        NSUInteger nb = (NSUInteger)n * sizeof(float);
        float zero[3] = {0.0f, 0.0f, 0.0f};
        unsigned int un = (unsigned int)n;
        id<MTLBuffer> bPred = make_buf_ro(gMDevice, predictions, nb);
        id<MTLBuffer> bTgt = make_buf_ro(gMDevice, targets, nb);
        id<MTLBuffer> bOut = [gMDevice newBufferWithBytes:zero length:sizeof(zero)
                                                   options:MTLResourceStorageModeShared];
        if (!bPred || !bTgt || !bOut) return -1;

        id<MTLCommandBuffer> cb = begin_cb();
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO_f1_counts];
        [enc setBuffer:bPred offset:0 atIndex:0];
        [enc setBuffer:bTgt offset:0 atIndex:1];
        [enc setBuffer:bOut offset:0 atIndex:2];
        [enc setBytes:&un length:sizeof(unsigned int) atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n,1,1)
        threadsPerThreadgroup:MTLSizeMake(gPSO_f1_counts.threadExecutionWidth,1,1)];
        [enc endEncoding];
        commit_wait(cb);
        memcpy(out, [bOut contents], 3 * sizeof(float));
        return 0;
    }
}
