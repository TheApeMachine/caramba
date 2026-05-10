#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "convolution.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Static globals — initialized once via metal_conv_init
// ---------------------------------------------------------------------------

static id<MTLDevice>               gConvDevice       = nil;
static id<MTLCommandQueue>         gConvQueue        = nil;
static id<MTLComputePipelineState> gPSO_conv1d       = nil;
static id<MTLComputePipelineState> gPSO_conv2d       = nil;
static id<MTLComputePipelineState> gPSO_conv3d       = nil;
static id<MTLComputePipelineState> gPSO_convt2d      = nil;

// ---------------------------------------------------------------------------

static id<MTLComputePipelineState> conv_make_pso(id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    id<MTLComputePipelineState> pso = [gConvDevice newComputePipelineStateWithFunction:fn error:&err];
    return pso;
}

int metal_conv_init(const char* metallib_path) {
    @autoreleasepool {
        gConvDevice = MTLCreateSystemDefaultDevice();
        if (!gConvDevice) return -1;

        gConvQueue = [gConvDevice newCommandQueue];
        if (!gConvQueue) return -1;

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError* err   = nil;
        id<MTLLibrary> lib = [gConvDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!lib) return -1;

        gPSO_conv1d  = conv_make_pso(lib, @"conv1d_kernel");
        gPSO_conv2d  = conv_make_pso(lib, @"conv2d_kernel");
        gPSO_conv3d  = conv_make_pso(lib, @"conv3d_kernel");
        gPSO_convt2d = conv_make_pso(lib, @"conv_transpose2d_kernel");

        if (!gPSO_conv1d || !gPSO_conv2d || !gPSO_conv3d || !gPSO_convt2d) return -1;
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Generic dispatch with src, dst, params, weight, bias buffers
// ---------------------------------------------------------------------------

typedef struct {
    const float*  src;
    NSUInteger    src_bytes;
    float*        dst;
    NSUInteger    dst_bytes;
    const void*   params;
    NSUInteger    params_bytes;
    const float*  weight;
    NSUInteger    weight_bytes;
    const float*  bias;
    NSUInteger    bias_bytes;
    int           grid_n;
} ConvDispatchArgs;

static int conv_dispatch(id<MTLComputePipelineState> pso, const ConvDispatchArgs* a)
{
    @autoreleasepool {
        vm_size_t page = getpagesize();

        // Helper lambda-like block to make an MTLBuffer from host ptr
        #define MAKE_BUF_RO(ptr, nbytes) \
            (((uintptr_t)(ptr) % page) == 0 \
                ? [gConvDevice newBufferWithBytesNoCopy:(void*)(ptr) length:(nbytes) \
                               options:MTLResourceStorageModeShared deallocator:nil] \
                : [gConvDevice newBufferWithBytes:(ptr) length:(nbytes) \
                               options:MTLResourceStorageModeShared])

        id<MTLBuffer> bufSrc    = MAKE_BUF_RO(a->src,    a->src_bytes);
        id<MTLBuffer> bufWeight = MAKE_BUF_RO(a->weight, a->weight_bytes);
        id<MTLBuffer> bufBias   = MAKE_BUF_RO(a->bias,   a->bias_bytes);
        #undef MAKE_BUF_RO

        id<MTLBuffer> bufDst;
        if (((uintptr_t)a->dst % page) == 0) {
            bufDst = [gConvDevice newBufferWithBytesNoCopy:a->dst
                                                   length:a->dst_bytes
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        } else {
            bufDst = [gConvDevice newBufferWithLength:a->dst_bytes
                                             options:MTLResourceStorageModeShared];
        }

        if (!bufSrc || !bufDst || !bufWeight || !bufBias) return -1;

        id<MTLCommandBuffer>        cb  = [gConvQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:bufSrc    offset:0 atIndex:0];
        [enc setBuffer:bufDst    offset:0 atIndex:1];
        [enc setBytes:a->params  length:a->params_bytes atIndex:2];
        [enc setBuffer:bufWeight offset:0 atIndex:3];
        [enc setBuffer:bufBias   offset:0 atIndex:4];

        NSUInteger tw  = pso.threadExecutionWidth;
        MTLSize threads     = MTLSizeMake((NSUInteger)a->grid_n, 1, 1);
        MTLSize threadgroup = MTLSizeMake(tw, 1, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (((uintptr_t)a->dst % page) != 0) {
            memcpy(a->dst, [bufDst contents], a->dst_bytes);
        }
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Params structs (must match Metal shader layout byte-for-byte)
// ---------------------------------------------------------------------------

typedef struct { int N,InC,L,OutC,K,stride,pad,dilation,groups,L_out; } MetalConv1dP;
typedef struct { int N,InC,H,W,OutC,KH,KW,sH,sW,pH,pW,dH,dW,groups,Hout,Wout; } MetalConv2dP;
typedef struct { int N,InC,D,H,W,OutC,KD,KH,KW,sD,sH,sW,pD,pH,pW,dD,dH,dW,groups,Dout,Hout,Wout; } MetalConv3dP;
typedef struct { int N,InC,H,W,OutC,KH,KW,sH,sW,pH,pW,dH,dW,groups,Hout,Wout; } MetalConvT2dP;

// ---------------------------------------------------------------------------

int metal_conv1d(
    const float* x, float* dst,
    int N, int InC, int L,
    int OutC, int K,
    int stride, int pad, int dilation, int groups,
    int L_out,
    const float* weight, const float* bias)
{
    MetalConv1dP p = { N, InC, L, OutC, K, stride, pad, dilation, groups, L_out };
    ConvDispatchArgs a = {
        .src         = x,
        .src_bytes   = (NSUInteger)(N*InC*L) * sizeof(float),
        .dst         = dst,
        .dst_bytes   = (NSUInteger)(N*OutC*L_out) * sizeof(float),
        .params      = &p,
        .params_bytes= sizeof(p),
        .weight      = weight,
        .weight_bytes= (NSUInteger)(OutC * (InC/groups) * K) * sizeof(float),
        .bias        = bias,
        .bias_bytes  = (NSUInteger)OutC * sizeof(float),
        .grid_n      = N * OutC * L_out,
    };
    return conv_dispatch(gPSO_conv1d, &a);
}

int metal_conv2d(
    const float* x, float* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const float* weight, const float* bias)
{
    MetalConv2dP p = { N,InC,H,W,OutC,KH,KW,sH,sW,pH,pW,dH,dW,groups,Hout,Wout };
    ConvDispatchArgs a = {
        .src         = x,
        .src_bytes   = (NSUInteger)(N*InC*H*W) * sizeof(float),
        .dst         = dst,
        .dst_bytes   = (NSUInteger)(N*OutC*Hout*Wout) * sizeof(float),
        .params      = &p,
        .params_bytes= sizeof(p),
        .weight      = weight,
        .weight_bytes= (NSUInteger)(OutC * (InC/groups) * KH * KW) * sizeof(float),
        .bias        = bias,
        .bias_bytes  = (NSUInteger)OutC * sizeof(float),
        .grid_n      = N * OutC * Hout * Wout,
    };
    return conv_dispatch(gPSO_conv2d, &a);
}

int metal_conv3d(
    const float* x, float* dst,
    int N, int InC, int D, int H, int W,
    int OutC, int KD, int KH, int KW,
    int sD, int sH, int sW, int pD, int pH, int pW,
    int dD, int dH, int dW, int groups,
    int Dout, int Hout, int Wout,
    const float* weight, const float* bias)
{
    MetalConv3dP p = { N,InC,D,H,W,OutC,KD,KH,KW,sD,sH,sW,pD,pH,pW,dD,dH,dW,groups,Dout,Hout,Wout };
    ConvDispatchArgs a = {
        .src         = x,
        .src_bytes   = (NSUInteger)(N*InC*D*H*W) * sizeof(float),
        .dst         = dst,
        .dst_bytes   = (NSUInteger)(N*OutC*Dout*Hout*Wout) * sizeof(float),
        .params      = &p,
        .params_bytes= sizeof(p),
        .weight      = weight,
        .weight_bytes= (NSUInteger)(OutC * (InC/groups) * KD * KH * KW) * sizeof(float),
        .bias        = bias,
        .bias_bytes  = (NSUInteger)OutC * sizeof(float),
        .grid_n      = N * OutC * Dout * Hout * Wout,
    };
    return conv_dispatch(gPSO_conv3d, &a);
}

int metal_conv_transpose2d(
    const float* x, float* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const float* weight, const float* bias)
{
    // Pre-fill dst with bias values before launching the scatter kernel.
    int outTotal = N * OutC * Hout * Wout;
    for (int ni = 0; ni < N; ni++) {
        for (int oc = 0; oc < OutC; oc++) {
            float b = bias[oc];
            int base = ni * OutC * Hout * Wout + oc * Hout * Wout;
            for (int i = 0; i < Hout * Wout; i++) {
                dst[base + i] = b;
            }
        }
    }

    MetalConvT2dP p = { N,InC,H,W,OutC,KH,KW,sH,sW,pH,pW,dH,dW,groups,Hout,Wout };
    ConvDispatchArgs a = {
        .src         = x,
        .src_bytes   = (NSUInteger)(N*InC*H*W) * sizeof(float),
        .dst         = dst,
        .dst_bytes   = (NSUInteger)outTotal * sizeof(float),
        .params      = &p,
        .params_bytes= sizeof(p),
        .weight      = weight,
        .weight_bytes= (NSUInteger)(InC * (OutC/groups) * KH * KW) * sizeof(float),
        .bias        = bias,
        .bias_bytes  = (NSUInteger)OutC * sizeof(float),
        .grid_n      = N * InC * H * W,
    };
    return conv_dispatch(gPSO_convt2d, &a);
}
