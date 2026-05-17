#include <stddef.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "causal.h"

extern id<MTLDevice> gCDevice;
extern id<MTLCommandQueue> gCQueue;
extern id<MTLComputePipelineState> gPSO_dag_full_prep;
extern id<MTLComputePipelineState> gPSO_dag_sigma2;
extern id<MTLComputePipelineState> gPSO_dag_full_score;
extern int gCInited;
extern dispatch_queue_t gCSerial;

void c_ensure_serial(void);
int c_wait(id<MTLCommandBuffer> cb);
void tensor_dispatch_ols(
	id<MTLCommandBuffer> commandBuffer,
	id<MTLBuffer> features,
	id<MTLBuffer> target,
	id<MTLBuffer> beta,
	id<MTLBuffer> errorBuffer,
	int samples,
	int featureCount,
	int outputCount,
	float ridge,
	id<MTLBuffer> featureCovariance,
	id<MTLBuffer> inverse,
	id<MTLBuffer> featureTarget,
	id<MTLBuffer> work);

int metal_causal_dag_markov_tensor(
	const void *X,
	const void *adj,
	void *log_prob,
	int T,
	int N)
{
	if (!gCInited) return -3;
	if (!X || !adj || !log_prob || T <= 0 || N <= 0) return -1;

	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		int featureCount = N + 1;
		size_t designBytes = (size_t)T * (size_t)featureCount * sizeof(float);
		size_t sampleBytes = (size_t)T * sizeof(float);
		size_t betaBytes = (size_t)featureCount * sizeof(float);
		size_t allBetaBytes = (size_t)N * (size_t)featureCount * sizeof(float);
		size_t squareBytes = (size_t)featureCount * (size_t)featureCount * sizeof(float);
		size_t workBytes = ((size_t)featureCount * (size_t)featureCount + 2 * (size_t)featureCount) * sizeof(float);

		id<MTLBuffer> observations = (__bridge id)((void *)X);
		id<MTLBuffer> adjacency = (__bridge id)((void *)adj);
		id<MTLBuffer> output = (__bridge id)log_prob;
		id<MTLBuffer> design = [gCDevice newBufferWithLength:designBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> nodeValues = [gCDevice newBufferWithLength:sampleBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> nodeBeta = [gCDevice newBufferWithLength:betaBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> betas = [gCDevice newBufferWithLength:allBetaBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> sigma2 = [gCDevice newBufferWithLength:(size_t)N * sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> featureCovariance = [gCDevice newBufferWithLength:squareBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> inverse = [gCDevice newBufferWithLength:squareBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> featureTarget = [gCDevice newBufferWithLength:betaBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> work = [gCDevice newBufferWithLength:workBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> errorBuffer = [gCDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];

		if (!observations || !adjacency || !output || !design || !nodeValues ||
		    !nodeBeta || !betas || !sigma2 || !featureCovariance || !inverse ||
		    !featureTarget || !work || !errorBuffer) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer> commandBuffer = [gCQueue commandBuffer];
		float ridge = 1e-10f;

		for (int nodeIndex = 0; nodeIndex < N; nodeIndex++) {
			id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
			[encoder setComputePipelineState:gPSO_dag_full_prep];
			[encoder setBuffer:observations offset:0 atIndex:0];
			[encoder setBuffer:adjacency offset:0 atIndex:1];
			[encoder setBuffer:design offset:0 atIndex:2];
			[encoder setBuffer:nodeValues offset:0 atIndex:3];
			[encoder setBytes:&T length:sizeof(T) atIndex:4];
			[encoder setBytes:&N length:sizeof(N) atIndex:5];
			[encoder setBytes:&nodeIndex length:sizeof(nodeIndex) atIndex:6];
			[encoder dispatchThreads:MTLSizeMake(1, 1, 1)
			 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
			[encoder endEncoding];

			tensor_dispatch_ols(
				commandBuffer,
				design,
				nodeValues,
				nodeBeta,
				errorBuffer,
				T,
				featureCount,
				1,
				ridge,
				featureCovariance,
				inverse,
				featureTarget,
				work);

			encoder = [commandBuffer computeCommandEncoder];
			[encoder setComputePipelineState:gPSO_dag_sigma2];
			[encoder setBuffer:design offset:0 atIndex:0];
			[encoder setBuffer:nodeValues offset:0 atIndex:1];
			[encoder setBuffer:nodeBeta offset:0 atIndex:2];
			[encoder setBuffer:sigma2 offset:0 atIndex:3];
			[encoder setBytes:&T length:sizeof(T) atIndex:4];
			[encoder setBytes:&featureCount length:sizeof(featureCount) atIndex:5];
			[encoder setBytes:&nodeIndex length:sizeof(nodeIndex) atIndex:6];
			[encoder dispatchThreads:MTLSizeMake(1, 1, 1)
			 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
			[encoder endEncoding];

			id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
			[blit copyFromBuffer:nodeBeta
			 sourceOffset:0
			 toBuffer:betas
			 destinationOffset:(NSUInteger)nodeIndex * (NSUInteger)featureCount * sizeof(float)
			 size:betaBytes];
			[blit endEncoding];
		}

		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		[encoder setComputePipelineState:gPSO_dag_full_score];
		[encoder setBuffer:observations offset:0 atIndex:0];
		[encoder setBuffer:adjacency offset:0 atIndex:1];
		[encoder setBuffer:betas offset:0 atIndex:2];
		[encoder setBuffer:sigma2 offset:0 atIndex:3];
		[encoder setBuffer:output offset:0 atIndex:4];
		[encoder setBytes:&T length:sizeof(T) atIndex:5];
		[encoder setBytes:&N length:sizeof(N) atIndex:6];
		[encoder dispatchThreads:MTLSizeMake(T, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
		[encoder endEncoding];

		rc = c_wait(commandBuffer);
	});

	return rc;
}
