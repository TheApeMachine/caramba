#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "embedding.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Static globals — initialized once via metal_embedding_init
// ---------------------------------------------------------------------------

static id<MTLDevice>               gEmbDevice = nil;
static id<MTLCommandQueue>         gEmbQueue  = nil;
static id<MTLComputePipelineState> gPSO_token_embedding = nil;

// ---------------------------------------------------------------------------

static id<MTLComputePipelineState> make_emb_pso(id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    return [gEmbDevice newComputePipelineStateWithFunction:fn error:&err];
}

int metal_embedding_init(const char* metallib_path) {
    @autoreleasepool {
        gEmbDevice = MTLCreateSystemDefaultDevice();
        if (!gEmbDevice) return -1;

        gEmbQueue = [gEmbDevice newCommandQueue];
        if (!gEmbQueue) return -1;

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError* err   = nil;
        id<MTLLibrary> lib = [gEmbDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!lib) return -1;

        gPSO_token_embedding = make_emb_pso(lib, @"token_embedding_kernel");
        if (!gPSO_token_embedding) return -1;

        return 0;
    }
}

// ---------------------------------------------------------------------------

int metal_token_embedding(
    const float* tokens,
    float*       out,
    const float* weight,
    int          batch_seq,
    int          d_model,
    int          vocab_size)
{
    @autoreleasepool {
        (void)vocab_size; // not used at dispatch level
        int total = batch_seq * d_model;

        NSUInteger weight_bytes = (NSUInteger)(vocab_size * d_model) * sizeof(float);
        NSUInteger token_bytes  = (NSUInteger)batch_seq * sizeof(float);
        NSUInteger out_bytes    = (NSUInteger)total * sizeof(float);

        vm_size_t page = getpagesize();

        // Weight buffer
        id<MTLBuffer> bufWeight;
        if (((uintptr_t)weight % page) == 0) {
            bufWeight = [gEmbDevice newBufferWithBytesNoCopy:(void*)weight
                                                     length:weight_bytes
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
        } else {
            bufWeight = [gEmbDevice newBufferWithBytes:weight length:weight_bytes
                                              options:MTLResourceStorageModeShared];
        }

        // Token buffer
        id<MTLBuffer> bufTokens;
        if (((uintptr_t)tokens % page) == 0) {
            bufTokens = [gEmbDevice newBufferWithBytesNoCopy:(void*)tokens
                                                     length:token_bytes
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
        } else {
            bufTokens = [gEmbDevice newBufferWithBytes:tokens length:token_bytes
                                              options:MTLResourceStorageModeShared];
        }

        // Output buffer (always writeable — do not NoCopy the caller's memory
        // unless it happens to be page-aligned; fall back to copy-back pattern)
        id<MTLBuffer> bufOut;
        BOOL out_nocopy = (((uintptr_t)out % page) == 0);
        if (out_nocopy) {
            bufOut = [gEmbDevice newBufferWithBytesNoCopy:out
                                                   length:out_bytes
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        } else {
            bufOut = [gEmbDevice newBufferWithLength:out_bytes
                                            options:MTLResourceStorageModeShared];
        }

        if (!bufWeight || !bufTokens || !bufOut) return -1;

        id<MTLCommandBuffer>         cb  = [gEmbQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:gPSO_token_embedding];
        [enc setBuffer:bufWeight offset:0 atIndex:0];
        [enc setBuffer:bufTokens offset:0 atIndex:1];
        [enc setBuffer:bufOut    offset:0 atIndex:2];
        int dm = d_model;
        [enc setBytes:&dm length:sizeof(int) atIndex:3];

        NSUInteger tw  = gPSO_token_embedding.threadExecutionWidth;
        NSUInteger tgs = tw;
        MTLSize threads     = MTLSizeMake((NSUInteger)total, 1, 1);
        MTLSize threadgroup = MTLSizeMake(tgs, 1, 1);

        [enc dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (!out_nocopy) {
            memcpy(out, [bufOut contents], out_bytes);
        }

        return 0;
    }
}
