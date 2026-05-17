#include <stddef.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "causal.h"

extern id<MTLDevice> gCDevice;
extern id<MTLCommandQueue> gCQueue;
extern id<MTLComputePipelineState> gPSO_chol_inv;
extern id<MTLComputePipelineState> gPSO_matvec;
extern id<MTLComputePipelineState> gPSO_matmul;
extern id<MTLComputePipelineState> gPSO_docalc_full_extract;
extern id<MTLComputePipelineState> gPSO_docalc_full_assemble;
extern int gCInited;
extern dispatch_queue_t gCSerial;

void c_ensure_serial(void);
int c_wait(id<MTLCommandBuffer> cb);

int metal_causal_do_calculus_tensor(
	const void *cov,
	const void *mask,
	const void *values,
	void *out,
	int N)
{
	if (!gCInited) return -3;
	if (!cov || !mask || !values || !out || N <= 0) return -1;

	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		size_t matrixBytes = (size_t)N * (size_t)N * sizeof(float);
		size_t vectorBytes = (size_t)N * sizeof(float);
		size_t workBytes = ((size_t)N * (size_t)N + 2 * (size_t)N) * sizeof(float);

		id<MTLBuffer> covariance = (__bridge id)((void *)cov);
		id<MTLBuffer> interventionMask = (__bridge id)((void *)mask);
		id<MTLBuffer> interventionValues = (__bridge id)((void *)values);
		id<MTLBuffer> output = (__bridge id)out;
		id<MTLBuffer> sigII = [gCDevice newBufferWithLength:matrixBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> sigFI = [gCDevice newBufferWithLength:matrixBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> sigFF = [gCDevice newBufferWithLength:matrixBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> sigIF = [gCDevice newBufferWithLength:matrixBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> xInt = [gCDevice newBufferWithLength:vectorBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> invII = [gCDevice newBufferWithLength:matrixBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> invX = [gCDevice newBufferWithLength:vectorBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> delta = [gCDevice newBufferWithLength:vectorBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> invIISigIF = [gCDevice newBufferWithLength:matrixBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> correction = [gCDevice newBufferWithLength:matrixBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> work = [gCDevice newBufferWithLength:workBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> errorBuffer = [gCDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];

		if (!covariance || !interventionMask || !interventionValues || !output ||
		    !sigII || !sigFI || !sigFF || !sigIF || !xInt || !invII ||
		    !invX || !delta || !invIISigIF || !correction || !work || !errorBuffer) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer> commandBuffer = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		[encoder setComputePipelineState:gPSO_docalc_full_extract];
		[encoder setBuffer:covariance offset:0 atIndex:0];
		[encoder setBuffer:interventionMask offset:0 atIndex:1];
		[encoder setBuffer:interventionValues offset:0 atIndex:2];
		[encoder setBuffer:sigII offset:0 atIndex:3];
		[encoder setBuffer:sigFI offset:0 atIndex:4];
		[encoder setBuffer:sigFF offset:0 atIndex:5];
		[encoder setBuffer:sigIF offset:0 atIndex:6];
		[encoder setBuffer:xInt offset:0 atIndex:7];
		[encoder setBytes:&N length:sizeof(N) atIndex:8];
		[encoder dispatchThreads:MTLSizeMake(1, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[encoder endEncoding];

		encoder = [commandBuffer computeCommandEncoder];
		[encoder setComputePipelineState:gPSO_chol_inv];
		[encoder setBuffer:sigII offset:0 atIndex:0];
		[encoder setBuffer:invII offset:0 atIndex:1];
		[encoder setBuffer:work offset:0 atIndex:2];
		[encoder setBuffer:errorBuffer offset:0 atIndex:3];
		[encoder setBytes:&N length:sizeof(N) atIndex:4];
		[encoder dispatchThreads:MTLSizeMake(1, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[encoder endEncoding];

		encoder = [commandBuffer computeCommandEncoder];
		[encoder setComputePipelineState:gPSO_matvec];
		[encoder setBuffer:invX offset:0 atIndex:0];
		[encoder setBuffer:invII offset:0 atIndex:1];
		[encoder setBuffer:xInt offset:0 atIndex:2];
		[encoder setBytes:&N length:sizeof(N) atIndex:3];
		[encoder setBytes:&N length:sizeof(N) atIndex:4];
		[encoder dispatchThreads:MTLSizeMake(N, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
		[encoder endEncoding];

		encoder = [commandBuffer computeCommandEncoder];
		[encoder setComputePipelineState:gPSO_matvec];
		[encoder setBuffer:delta offset:0 atIndex:0];
		[encoder setBuffer:sigFI offset:0 atIndex:1];
		[encoder setBuffer:invX offset:0 atIndex:2];
		[encoder setBytes:&N length:sizeof(N) atIndex:3];
		[encoder setBytes:&N length:sizeof(N) atIndex:4];
		[encoder dispatchThreads:MTLSizeMake(N, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
		[encoder endEncoding];

		encoder = [commandBuffer computeCommandEncoder];
		[encoder setComputePipelineState:gPSO_matmul];
		[encoder setBuffer:invII offset:0 atIndex:0];
		[encoder setBuffer:sigIF offset:0 atIndex:1];
		[encoder setBuffer:invIISigIF offset:0 atIndex:2];
		[encoder setBytes:&N length:sizeof(N) atIndex:3];
		[encoder setBytes:&N length:sizeof(N) atIndex:4];
		[encoder setBytes:&N length:sizeof(N) atIndex:5];
		[encoder dispatchThreads:MTLSizeMake(N, N, 1)
		 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
		[encoder endEncoding];

		encoder = [commandBuffer computeCommandEncoder];
		[encoder setComputePipelineState:gPSO_matmul];
		[encoder setBuffer:sigFI offset:0 atIndex:0];
		[encoder setBuffer:invIISigIF offset:0 atIndex:1];
		[encoder setBuffer:correction offset:0 atIndex:2];
		[encoder setBytes:&N length:sizeof(N) atIndex:3];
		[encoder setBytes:&N length:sizeof(N) atIndex:4];
		[encoder setBytes:&N length:sizeof(N) atIndex:5];
		[encoder dispatchThreads:MTLSizeMake(N, N, 1)
		 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
		[encoder endEncoding];

		encoder = [commandBuffer computeCommandEncoder];
		[encoder setComputePipelineState:gPSO_docalc_full_assemble];
		[encoder setBuffer:output offset:0 atIndex:0];
		[encoder setBuffer:interventionMask offset:0 atIndex:1];
		[encoder setBuffer:interventionValues offset:0 atIndex:2];
		[encoder setBuffer:delta offset:0 atIndex:3];
		[encoder setBuffer:sigFF offset:0 atIndex:4];
		[encoder setBuffer:correction offset:0 atIndex:5];
		[encoder setBytes:&N length:sizeof(N) atIndex:6];
		[encoder dispatchThreads:MTLSizeMake(1, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[encoder endEncoding];

		rc = c_wait(commandBuffer);
	});

	return rc;
}
