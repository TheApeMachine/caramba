#include <stddef.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "causal.h"

extern id<MTLDevice> gCDevice;
extern id<MTLCommandQueue> gCQueue;
extern id<MTLComputePipelineState> gPSO_counterfactual;
extern id<MTLComputePipelineState> gPSO_frontdoor_sort_pad;
extern id<MTLComputePipelineState> gPSO_frontdoor_sort_step;
extern id<MTLComputePipelineState> gPSO_frontdoor_boundaries;
extern id<MTLComputePipelineState> gPSO_frontdoor_assign;
extern id<MTLComputePipelineState> gPSO_frontdoor_accumulate;
extern id<MTLComputePipelineState> gPSO_frontdoor_normalize;
extern id<MTLComputePipelineState> gPSO_frontdoor_effect;
extern int gCInited;
extern dispatch_queue_t gCSerial;

void c_ensure_serial(void);
int c_wait(id<MTLCommandBuffer> cb);

static int tensor_next_power_of_two(int value) {
	int capacity = 1;

	while (capacity < value) {
		capacity <<= 1;
	}

	return capacity;
}

static void tensor_encode_frontdoor_sort(
	id<MTLCommandBuffer> commandBuffer,
	id<MTLBuffer> values,
	id<MTLBuffer> sorted,
	int samples,
	int paddedSamples)
{
	id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
	[encoder setComputePipelineState:gPSO_frontdoor_sort_pad];
	[encoder setBuffer:values offset:0 atIndex:0];
	[encoder setBuffer:sorted offset:0 atIndex:1];
	[encoder setBytes:&samples length:sizeof(samples) atIndex:2];
	[encoder setBytes:&paddedSamples length:sizeof(paddedSamples) atIndex:3];
	[encoder dispatchThreads:MTLSizeMake((NSUInteger)paddedSamples, 1, 1)
	 threadsPerThreadgroup:MTLSizeMake(gPSO_frontdoor_sort_pad.threadExecutionWidth, 1, 1)];
	[encoder endEncoding];

	for (uint mergeWidth = 2; mergeWidth <= (uint)paddedSamples; mergeWidth <<= 1) {
		for (uint compareDistance = mergeWidth >> 1; compareDistance > 0; compareDistance >>= 1) {
			encoder = [commandBuffer computeCommandEncoder];
			[encoder setComputePipelineState:gPSO_frontdoor_sort_step];
			[encoder setBuffer:sorted offset:0 atIndex:0];
			[encoder setBytes:&compareDistance length:sizeof(compareDistance) atIndex:1];
			[encoder setBytes:&mergeWidth length:sizeof(mergeWidth) atIndex:2];
			uint padded = (uint)paddedSamples;
			[encoder setBytes:&padded length:sizeof(padded) atIndex:3];
			[encoder dispatchThreads:MTLSizeMake((NSUInteger)paddedSamples, 1, 1)
			 threadsPerThreadgroup:MTLSizeMake(gPSO_frontdoor_sort_step.threadExecutionWidth, 1, 1)];
			[encoder endEncoding];
		}
	}
}

int metal_causal_counterfactual_tensor(
	const void *X_obs,
	const void *Y_obs,
	const void *beta,
	const void *X_cf,
	void *out,
	int N,
	int N_cf)
{
	if (!gCInited) return -3;
	if (!X_obs || !Y_obs || !beta || !X_cf || !out || N <= 0 || N_cf <= 0) return -1;

	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		id<MTLBuffer> bX = (__bridge id)((void *)X_obs);
		id<MTLBuffer> bY = (__bridge id)((void *)Y_obs);
		id<MTLBuffer> bBeta = (__bridge id)((void *)beta);
		id<MTLBuffer> bXcf = (__bridge id)((void *)X_cf);
		id<MTLBuffer> bOut = (__bridge id)out;
		if (!bX || !bY || !bBeta || !bXcf || !bOut) { rc = -4; return; }

		id<MTLCommandBuffer> cb = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_counterfactual];
		[enc setBuffer:bX offset:0 atIndex:0];
		[enc setBuffer:bY offset:0 atIndex:1];
		[enc setBuffer:bBeta offset:0 atIndex:2];
		[enc setBuffer:bXcf offset:0 atIndex:3];
		[enc setBuffer:bOut offset:0 atIndex:4];
		[enc setBytes:&N length:sizeof(N) atIndex:5];
		[enc setBytes:&N_cf length:sizeof(N_cf) atIndex:6];
		NSUInteger totalThreads = (NSUInteger)N * (NSUInteger)N_cf;
		[enc dispatchThreads:MTLSizeMake(totalThreads, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(gPSO_counterfactual.threadExecutionWidth, 1, 1)];
		[enc endEncoding];

		rc = c_wait(cb);
	});

	return rc;
}

int metal_causal_frontdoor_tensor(
	const void *X,
	const void *M,
	const void *Y,
	void *effect,
	int T,
	int nx,
	int nm)
{
	if (!gCInited) return -3;
	if (!X || !M || !Y || !effect || T <= 0 || nx <= 0 || nm <= 0) return -1;

	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		size_t xBoundaryBytes = (size_t)(nx > 1 ? nx - 1 : 1) * sizeof(float);
		size_t mBoundaryBytes = (size_t)(nm > 1 ? nm - 1 : 1) * sizeof(float);
		size_t xBytes = (size_t)nx * sizeof(float);
		size_t matrixBytes = (size_t)nx * (size_t)nm * sizeof(float);
		size_t binBytes = (size_t)T * sizeof(int);
		int paddedSamples = tensor_next_power_of_two(T);
		size_t sortedBytes = (size_t)paddedSamples * sizeof(float);

		id<MTLBuffer> bX = (__bridge id)((void *)X);
		id<MTLBuffer> bM = (__bridge id)((void *)M);
		id<MTLBuffer> bY = (__bridge id)((void *)Y);
		id<MTLBuffer> bEffect = (__bridge id)effect;
		id<MTLBuffer> bXSorted = nx > 1 ? [gCDevice newBufferWithLength:sortedBytes options:MTLResourceStorageModeShared] : nil;
		id<MTLBuffer> bMSorted = nm > 1 ? [gCDevice newBufferWithLength:sortedBytes options:MTLResourceStorageModeShared] : nil;
		id<MTLBuffer> bXBoundaries = [gCDevice newBufferWithLength:xBoundaryBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> bMBoundaries = [gCDevice newBufferWithLength:mBoundaryBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> bXBins = [gCDevice newBufferWithLength:binBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> bMBins = [gCDevice newBufferWithLength:binBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> bPX = [gCDevice newBufferWithLength:xBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> bCountX = [gCDevice newBufferWithLength:xBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> bMGivenX = [gCDevice newBufferWithLength:matrixBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> bEYGivenXM = [gCDevice newBufferWithLength:matrixBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> bCountXM = [gCDevice newBufferWithLength:matrixBytes options:MTLResourceStorageModeShared];

		if (!bX || !bM || !bY || !bEffect || !bXBoundaries || !bMBoundaries || !bXBins || !bMBins ||
		    !bPX || !bCountX || !bMGivenX || !bEYGivenXM || !bCountXM) {
			rc = -4;
			return;
		}
		if ((nx > 1 && !bXSorted) || (nm > 1 && !bMSorted)) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer> cb = [gCQueue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit fillBuffer:bPX range:NSMakeRange(0, xBytes) value:0];
		[blit fillBuffer:bCountX range:NSMakeRange(0, xBytes) value:0];
		[blit fillBuffer:bMGivenX range:NSMakeRange(0, matrixBytes) value:0];
		[blit fillBuffer:bEYGivenXM range:NSMakeRange(0, matrixBytes) value:0];
		[blit fillBuffer:bCountXM range:NSMakeRange(0, matrixBytes) value:0];
		[blit endEncoding];

		if (nx > 1) tensor_encode_frontdoor_sort(cb, bX, bXSorted, T, paddedSamples);
		if (nm > 1) tensor_encode_frontdoor_sort(cb, bM, bMSorted, T, paddedSamples);

		if (nx > 1) {
			id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
			[enc setComputePipelineState:gPSO_frontdoor_boundaries];
			[enc setBuffer:bXSorted offset:0 atIndex:0];
			[enc setBuffer:bXBoundaries offset:0 atIndex:1];
			[enc setBytes:&T length:sizeof(T) atIndex:2];
			[enc setBytes:&nx length:sizeof(nx) atIndex:3];
			[enc dispatchThreads:MTLSizeMake((NSUInteger)(nx - 1), 1, 1)
			 threadsPerThreadgroup:MTLSizeMake(gPSO_frontdoor_boundaries.threadExecutionWidth, 1, 1)];
			[enc endEncoding];
		}

		if (nm > 1) {
			id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
			[enc setComputePipelineState:gPSO_frontdoor_boundaries];
			[enc setBuffer:bMSorted offset:0 atIndex:0];
			[enc setBuffer:bMBoundaries offset:0 atIndex:1];
			[enc setBytes:&T length:sizeof(T) atIndex:2];
			[enc setBytes:&nm length:sizeof(nm) atIndex:3];
			[enc dispatchThreads:MTLSizeMake((NSUInteger)(nm - 1), 1, 1)
			 threadsPerThreadgroup:MTLSizeMake(gPSO_frontdoor_boundaries.threadExecutionWidth, 1, 1)];
			[enc endEncoding];
		}

		id<MTLComputeCommandEncoder> encX = [cb computeCommandEncoder];
		[encX setComputePipelineState:gPSO_frontdoor_assign];
		[encX setBuffer:bX offset:0 atIndex:0]; [encX setBuffer:bXBoundaries offset:0 atIndex:1];
		[encX setBuffer:bXBins offset:0 atIndex:2]; [encX setBytes:&T length:sizeof(T) atIndex:3];
		[encX setBytes:&nx length:sizeof(nx) atIndex:4];
		[encX dispatchThreads:MTLSizeMake((NSUInteger)T, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(gPSO_frontdoor_assign.threadExecutionWidth, 1, 1)];
		[encX endEncoding];

		id<MTLComputeCommandEncoder> encM = [cb computeCommandEncoder];
		[encM setComputePipelineState:gPSO_frontdoor_assign];
		[encM setBuffer:bM offset:0 atIndex:0]; [encM setBuffer:bMBoundaries offset:0 atIndex:1];
		[encM setBuffer:bMBins offset:0 atIndex:2]; [encM setBytes:&T length:sizeof(T) atIndex:3];
		[encM setBytes:&nm length:sizeof(nm) atIndex:4];
		[encM dispatchThreads:MTLSizeMake((NSUInteger)T, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(gPSO_frontdoor_assign.threadExecutionWidth, 1, 1)];
		[encM endEncoding];

		id<MTLComputeCommandEncoder> encA = [cb computeCommandEncoder];
		[encA setComputePipelineState:gPSO_frontdoor_accumulate];
		[encA setBuffer:bXBins offset:0 atIndex:0]; [encA setBuffer:bMBins offset:0 atIndex:1];
		[encA setBuffer:bY offset:0 atIndex:2]; [encA setBuffer:bPX offset:0 atIndex:3];
		[encA setBuffer:bCountX offset:0 atIndex:4]; [encA setBuffer:bMGivenX offset:0 atIndex:5];
		[encA setBuffer:bEYGivenXM offset:0 atIndex:6]; [encA setBuffer:bCountXM offset:0 atIndex:7];
		[encA setBytes:&T length:sizeof(T) atIndex:8]; [encA setBytes:&nx length:sizeof(nx) atIndex:9];
		[encA setBytes:&nm length:sizeof(nm) atIndex:10];
		[encA dispatchThreads:MTLSizeMake((NSUInteger)T, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(gPSO_frontdoor_accumulate.threadExecutionWidth, 1, 1)];
		[encA endEncoding];

		id<MTLComputeCommandEncoder> encN = [cb computeCommandEncoder];
		[encN setComputePipelineState:gPSO_frontdoor_normalize];
		[encN setBuffer:bPX offset:0 atIndex:0]; [encN setBuffer:bCountX offset:0 atIndex:1];
		[encN setBuffer:bMGivenX offset:0 atIndex:2]; [encN setBuffer:bEYGivenXM offset:0 atIndex:3];
		[encN setBuffer:bCountXM offset:0 atIndex:4]; [encN setBytes:&T length:sizeof(T) atIndex:5];
		[encN setBytes:&nx length:sizeof(nx) atIndex:6]; [encN setBytes:&nm length:sizeof(nm) atIndex:7];
		NSUInteger normalizeThreads = (NSUInteger)nx * (NSUInteger)nm;
		[encN dispatchThreads:MTLSizeMake(normalizeThreads, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(gPSO_frontdoor_normalize.threadExecutionWidth, 1, 1)];
		[encN endEncoding];

		id<MTLComputeCommandEncoder> encE = [cb computeCommandEncoder];
		[encE setComputePipelineState:gPSO_frontdoor_effect];
		[encE setBuffer:bPX offset:0 atIndex:0]; [encE setBuffer:bMGivenX offset:0 atIndex:1];
		[encE setBuffer:bEYGivenXM offset:0 atIndex:2]; [encE setBuffer:bEffect offset:0 atIndex:3];
		[encE setBytes:&nx length:sizeof(nx) atIndex:4]; [encE setBytes:&nm length:sizeof(nm) atIndex:5];
		[encE dispatchThreads:MTLSizeMake((NSUInteger)nx, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(gPSO_frontdoor_effect.threadExecutionWidth, 1, 1)];
		[encE endEncoding];

		rc = c_wait(cb);
	});

	return rc;
}
