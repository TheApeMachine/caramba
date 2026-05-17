#include <stddef.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "causal.h"

extern id<MTLDevice> gCDevice;
extern id<MTLCommandQueue> gCQueue;
extern id<MTLComputePipelineState> gPSO_ata;
extern id<MTLComputePipelineState> gPSO_atb;
extern id<MTLComputePipelineState> gPSO_matmul;
extern id<MTLComputePipelineState> gPSO_chol_inv;
extern id<MTLComputePipelineState> gPSO_backdoor_design;
extern id<MTLComputePipelineState> gPSO_backdoor_effect;
extern id<MTLComputePipelineState> gPSO_cate_split;
extern id<MTLComputePipelineState> gPSO_cate_effect_counted;
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
	id<MTLBuffer> work)
{
	id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
	[encoder setComputePipelineState:gPSO_ata];
	[encoder setBuffer:features offset:0 atIndex:0];
	[encoder setBuffer:featureCovariance offset:0 atIndex:1];
	[encoder setBytes:&samples length:sizeof(samples) atIndex:2];
	[encoder setBytes:&featureCount length:sizeof(featureCount) atIndex:3];
	[encoder setBytes:&ridge length:sizeof(ridge) atIndex:4];
	[encoder dispatchThreads:MTLSizeMake(featureCount, featureCount, 1)
	 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
	[encoder endEncoding];

	encoder = [commandBuffer computeCommandEncoder];
	[encoder setComputePipelineState:gPSO_chol_inv];
	[encoder setBuffer:featureCovariance offset:0 atIndex:0];
	[encoder setBuffer:inverse offset:0 atIndex:1];
	[encoder setBuffer:work offset:0 atIndex:2];
	[encoder setBuffer:errorBuffer offset:0 atIndex:3];
	[encoder setBytes:&featureCount length:sizeof(featureCount) atIndex:4];
	[encoder dispatchThreads:MTLSizeMake(1, 1, 1)
	 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
	[encoder endEncoding];

	encoder = [commandBuffer computeCommandEncoder];
	[encoder setComputePipelineState:gPSO_atb];
	[encoder setBuffer:features offset:0 atIndex:0];
	[encoder setBuffer:target offset:0 atIndex:1];
	[encoder setBuffer:featureTarget offset:0 atIndex:2];
	[encoder setBytes:&samples length:sizeof(samples) atIndex:3];
	[encoder setBytes:&featureCount length:sizeof(featureCount) atIndex:4];
	[encoder setBytes:&outputCount length:sizeof(outputCount) atIndex:5];
	[encoder dispatchThreads:MTLSizeMake(featureCount, outputCount, 1)
	 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
	[encoder endEncoding];

	encoder = [commandBuffer computeCommandEncoder];
	[encoder setComputePipelineState:gPSO_matmul];
	[encoder setBuffer:inverse offset:0 atIndex:0];
	[encoder setBuffer:featureTarget offset:0 atIndex:1];
	[encoder setBuffer:beta offset:0 atIndex:2];
	[encoder setBytes:&featureCount length:sizeof(featureCount) atIndex:3];
	[encoder setBytes:&featureCount length:sizeof(featureCount) atIndex:4];
	[encoder setBytes:&outputCount length:sizeof(outputCount) atIndex:5];
	[encoder dispatchThreads:MTLSizeMake(featureCount, outputCount, 1)
	 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
	[encoder endEncoding];
}

int metal_causal_backdoor_tensor(
	const void *Y,
	const void *X,
	const void *Z,
	void *effect,
	int T,
	int ny,
	int nx,
	int nz)
{
	if (!gCInited) return -3;
	if (!Y || !X || !effect || T <= 0 || ny <= 0 || nx <= 0 || nz < 0) return -1;
	if (nz > 0 && !Z) return -1;

	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		int featureCount = 1 + nx + nz;
		size_t designBytes = (size_t)T * (size_t)featureCount * sizeof(float);
		size_t squareBytes = (size_t)featureCount * (size_t)featureCount * sizeof(float);
		size_t betaBytes = (size_t)featureCount * (size_t)ny * sizeof(float);
		size_t workBytes = ((size_t)featureCount * (size_t)featureCount + 2 * (size_t)featureCount) * sizeof(float);

		id<MTLBuffer> outcome = (__bridge id)((void *)Y);
		id<MTLBuffer> treatment = (__bridge id)((void *)X);
		id<MTLBuffer> confounder = (__bridge id)((void *)Z);
		id<MTLBuffer> output = (__bridge id)effect;
		id<MTLBuffer> design = [gCDevice newBufferWithLength:designBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> featureCovariance = [gCDevice newBufferWithLength:squareBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> inverse = [gCDevice newBufferWithLength:squareBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> featureTarget = [gCDevice newBufferWithLength:betaBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> beta = [gCDevice newBufferWithLength:betaBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> work = [gCDevice newBufferWithLength:workBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> errorBuffer = [gCDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];

		if (!outcome || !treatment || !output || !design || !featureCovariance ||
		    !inverse || !featureTarget || !beta || !work || !errorBuffer) {
			rc = -4;
			return;
		}
		if (nz > 0 && !confounder) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer> commandBuffer = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		[encoder setComputePipelineState:gPSO_backdoor_design];
		[encoder setBuffer:treatment offset:0 atIndex:0];
		[encoder setBuffer:confounder offset:0 atIndex:1];
		[encoder setBuffer:design offset:0 atIndex:2];
		[encoder setBytes:&T length:sizeof(T) atIndex:3];
		[encoder setBytes:&nx length:sizeof(nx) atIndex:4];
		[encoder setBytes:&nz length:sizeof(nz) atIndex:5];
		[encoder dispatchThreads:MTLSizeMake(T, featureCount, 1)
		 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
		[encoder endEncoding];

		float ridge = 1e-10f;
		tensor_dispatch_ols(
			commandBuffer,
			design,
			outcome,
			beta,
			errorBuffer,
			T,
			featureCount,
			ny,
			ridge,
			featureCovariance,
			inverse,
			featureTarget,
			work);

		encoder = [commandBuffer computeCommandEncoder];
		[encoder setComputePipelineState:gPSO_backdoor_effect];
		[encoder setBuffer:beta offset:0 atIndex:0];
		[encoder setBuffer:output offset:0 atIndex:1];
		[encoder setBytes:&ny length:sizeof(ny) atIndex:2];
		[encoder setBytes:&nx length:sizeof(nx) atIndex:3];
		[encoder setBytes:&featureCount length:sizeof(featureCount) atIndex:4];
		[encoder dispatchThreads:MTLSizeMake(ny, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
		[encoder endEncoding];

		rc = c_wait(commandBuffer);
		if (rc != 0) return;
		if (((int *)[errorBuffer contents])[0] != 0) rc = -5;
	});

	return rc;
}

int metal_causal_cate_tensor(
	const void *X,
	const void *treatment,
	const void *Y,
	void *cate,
	int T,
	int nx)
{
	if (!gCInited) return -3;
	if (!X || !treatment || !Y || !cate || T <= 0 || nx <= 0) return -1;

	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		int featureCount = nx + 1;
		size_t featureBytes = (size_t)T * (size_t)featureCount * sizeof(float);
		size_t sampleBytes = (size_t)T * sizeof(float);
		size_t squareBytes = (size_t)featureCount * (size_t)featureCount * sizeof(float);
		size_t betaBytes = (size_t)featureCount * sizeof(float);
		size_t workBytes = ((size_t)featureCount * (size_t)featureCount + 2 * (size_t)featureCount) * sizeof(float);

		id<MTLBuffer> features = (__bridge id)((void *)X);
		id<MTLBuffer> treatmentValues = (__bridge id)((void *)treatment);
		id<MTLBuffer> outcome = (__bridge id)((void *)Y);
		id<MTLBuffer> output = (__bridge id)cate;
		id<MTLBuffer> treatedFeatures = [gCDevice newBufferWithLength:featureBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> treatedOutcome = [gCDevice newBufferWithLength:sampleBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> controlFeatures = [gCDevice newBufferWithLength:featureBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> controlOutcome = [gCDevice newBufferWithLength:sampleBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> counts = [gCDevice newBufferWithLength:2 * sizeof(int) options:MTLResourceStorageModeShared];
		id<MTLBuffer> featureCovariance = [gCDevice newBufferWithLength:squareBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> inverse = [gCDevice newBufferWithLength:squareBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> featureTarget = [gCDevice newBufferWithLength:betaBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> treatedBeta = [gCDevice newBufferWithLength:betaBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> controlBeta = [gCDevice newBufferWithLength:betaBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> work = [gCDevice newBufferWithLength:workBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> errorBuffer = [gCDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];

		if (!features || !treatmentValues || !outcome || !output || !treatedFeatures ||
		    !treatedOutcome || !controlFeatures || !controlOutcome || !counts ||
		    !featureCovariance || !inverse || !featureTarget || !treatedBeta ||
		    !controlBeta || !work || !errorBuffer) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer> commandBuffer = [gCQueue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
		[blit fillBuffer:treatedFeatures range:NSMakeRange(0, featureBytes) value:0];
		[blit fillBuffer:treatedOutcome range:NSMakeRange(0, sampleBytes) value:0];
		[blit fillBuffer:controlFeatures range:NSMakeRange(0, featureBytes) value:0];
		[blit fillBuffer:controlOutcome range:NSMakeRange(0, sampleBytes) value:0];
		[blit fillBuffer:counts range:NSMakeRange(0, 2 * sizeof(int)) value:0];
		[blit endEncoding];

		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		[encoder setComputePipelineState:gPSO_cate_split];
		[encoder setBuffer:features offset:0 atIndex:0];
		[encoder setBuffer:treatmentValues offset:0 atIndex:1];
		[encoder setBuffer:outcome offset:0 atIndex:2];
		[encoder setBuffer:treatedFeatures offset:0 atIndex:3];
		[encoder setBuffer:treatedOutcome offset:0 atIndex:4];
		[encoder setBuffer:controlFeatures offset:0 atIndex:5];
		[encoder setBuffer:controlOutcome offset:0 atIndex:6];
		[encoder setBuffer:counts offset:0 atIndex:7];
		[encoder setBytes:&T length:sizeof(T) atIndex:8];
		[encoder setBytes:&nx length:sizeof(nx) atIndex:9];
		[encoder dispatchThreads:MTLSizeMake(1, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[encoder endEncoding];

		float ridge = 1e-10f;
		tensor_dispatch_ols(
			commandBuffer,
			treatedFeatures,
			treatedOutcome,
			treatedBeta,
			errorBuffer,
			T,
			featureCount,
			1,
			ridge,
			featureCovariance,
			inverse,
			featureTarget,
			work);
		tensor_dispatch_ols(
			commandBuffer,
			controlFeatures,
			controlOutcome,
			controlBeta,
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
		[encoder setComputePipelineState:gPSO_cate_effect_counted];
		[encoder setBuffer:features offset:0 atIndex:0];
		[encoder setBuffer:treatedBeta offset:0 atIndex:1];
		[encoder setBuffer:controlBeta offset:0 atIndex:2];
		[encoder setBuffer:counts offset:0 atIndex:3];
		[encoder setBuffer:output offset:0 atIndex:4];
		[encoder setBytes:&T length:sizeof(T) atIndex:5];
		[encoder setBytes:&nx length:sizeof(nx) atIndex:6];
		[encoder dispatchThreads:MTLSizeMake(T, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
		[encoder endEncoding];

		rc = c_wait(commandBuffer);
	});

	return rc;
}

int metal_causal_iv_tensor(
	const void *Z,
	const void *X,
	const void *Y,
	void *beta_iv,
	int T,
	int nz,
	int nx,
	int ny)
{
	if (!gCInited) return -3;
	if (!Z || !X || !Y || !beta_iv || T <= 0 || nz <= 0 || nx <= 0 || ny <= 0) return -1;

	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		int maxFeatures = nz > nx ? nz : nx;
		int maxTargets = nx > ny ? nx : ny;
		size_t squareBytes = (size_t)maxFeatures * (size_t)maxFeatures * sizeof(float);
		size_t targetBytes = (size_t)maxFeatures * (size_t)maxTargets * sizeof(float);
		size_t workBytes = ((size_t)maxFeatures * (size_t)maxFeatures + 2 * (size_t)maxFeatures) * sizeof(float);
		size_t projectionBytes = (size_t)nz * (size_t)nx * sizeof(float);
		size_t projectedTreatmentBytes = (size_t)T * (size_t)nx * sizeof(float);

		id<MTLBuffer> instrument = (__bridge id)((void *)Z);
		id<MTLBuffer> treatment = (__bridge id)((void *)X);
		id<MTLBuffer> outcome = (__bridge id)((void *)Y);
		id<MTLBuffer> output = (__bridge id)beta_iv;
		id<MTLBuffer> featureCovariance = [gCDevice newBufferWithLength:squareBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> inverse = [gCDevice newBufferWithLength:squareBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> featureTarget = [gCDevice newBufferWithLength:targetBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> projection = [gCDevice newBufferWithLength:projectionBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> projectedTreatment = [gCDevice newBufferWithLength:projectedTreatmentBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> work = [gCDevice newBufferWithLength:workBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> errorBuffer = [gCDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];

		if (!instrument || !treatment || !outcome || !output || !featureCovariance ||
		    !inverse || !featureTarget || !projection || !projectedTreatment ||
		    !work || !errorBuffer) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer> commandBuffer = [gCQueue commandBuffer];
		float ridge = 1e-10f;
		tensor_dispatch_ols(
			commandBuffer,
			instrument,
			treatment,
			projection,
			errorBuffer,
			T,
			nz,
			nx,
			ridge,
			featureCovariance,
			inverse,
			featureTarget,
			work);

		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		[encoder setComputePipelineState:gPSO_matmul];
		[encoder setBuffer:instrument offset:0 atIndex:0];
		[encoder setBuffer:projection offset:0 atIndex:1];
		[encoder setBuffer:projectedTreatment offset:0 atIndex:2];
		[encoder setBytes:&T length:sizeof(T) atIndex:3];
		[encoder setBytes:&nz length:sizeof(nz) atIndex:4];
		[encoder setBytes:&nx length:sizeof(nx) atIndex:5];
		[encoder dispatchThreads:MTLSizeMake(T, nx, 1)
		 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
		[encoder endEncoding];

		tensor_dispatch_ols(
			commandBuffer,
			projectedTreatment,
			outcome,
			output,
			errorBuffer,
			T,
			nx,
			ny,
			ridge,
			featureCovariance,
			inverse,
			featureTarget,
			work);

		rc = c_wait(commandBuffer);
	});

	return rc;
}
