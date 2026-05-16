#include <math.h>
#include <stddef.h>
#include <string.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "causal.h"

id<MTLDevice> gCDevice = nil;
id<MTLCommandQueue> gCQueue = nil;
static id<MTLComputePipelineState> gPSO_axpy = nil;
static id<MTLComputePipelineState> gPSO_sub = nil;
static id<MTLComputePipelineState> gPSO_matvec = nil;
static id<MTLComputePipelineState> gPSO_dotatom = nil;
static id<MTLComputePipelineState> gPSO_ata = nil;
static id<MTLComputePipelineState> gPSO_atb = nil;
static id<MTLComputePipelineState> gPSO_matmul = nil;
static id<MTLComputePipelineState> gPSO_chol_inv = nil;

static id<MTLComputePipelineState> gPSO_docalc_extract = nil;
static id<MTLComputePipelineState> gPSO_docalc_assemble = nil;

static id<MTLComputePipelineState> gPSO_backdoor_design = nil;
static id<MTLComputePipelineState> gPSO_backdoor_effect = nil;

static id<MTLComputePipelineState> gPSO_cate_split = nil;
static id<MTLComputePipelineState> gPSO_cate_effect = nil;

id<MTLComputePipelineState> gPSO_counterfactual = nil;
id<MTLComputePipelineState> gPSO_frontdoor_sort_pad = nil;
id<MTLComputePipelineState> gPSO_frontdoor_sort_step = nil;
id<MTLComputePipelineState> gPSO_frontdoor_boundaries = nil;
id<MTLComputePipelineState> gPSO_frontdoor_assign = nil;
id<MTLComputePipelineState> gPSO_frontdoor_accumulate = nil;
id<MTLComputePipelineState> gPSO_frontdoor_normalize = nil;
id<MTLComputePipelineState> gPSO_frontdoor_effect = nil;

static id<MTLComputePipelineState> gPSO_dag_prep = nil;
static id<MTLComputePipelineState> gPSO_dag_sigma2 = nil;
static id<MTLComputePipelineState> gPSO_dag_score = nil;

int gCInited = 0;
dispatch_queue_t gCSerial = NULL;

void c_ensure_serial(void) {
	static dispatch_once_t onceToken;
	dispatch_once(&onceToken, ^{
		gCSerial = dispatch_queue_create("com.caramba.metal.causal", DISPATCH_QUEUE_SERIAL);
	});
}

static id<MTLComputePipelineState> c_make_pso(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
	NSError *err = nil;
	id<MTLFunction> fn = [library newFunctionWithName:name];
	if (!fn) return nil;
	return [device newComputePipelineStateWithFunction:fn error:&err];
}

int c_wait(id<MTLCommandBuffer> cb) {
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	if (!done) return -1;
	[cb addCompletedHandler:^(id<MTLCommandBuffer> _) {
		dispatch_semaphore_signal(done);
	}];
	[cb commit];
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
	if (cb.error != nil) return -3;
	return 0;
}

static int next_power_of_two_int(int value) {
	int capacity = 1;

	while (capacity < value) {
		capacity <<= 1;
	}

	return capacity;
}

static void encode_frontdoor_sort(
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

int metal_causal_init(const char *metallib_path) {
	if (!metallib_path || metallib_path[0] == '\0') return -1;
	c_ensure_serial();
	__block int result = 0;
	dispatch_sync(gCSerial, ^{
		if (gCInited) return;
		gCDevice = MTLCreateSystemDefaultDevice();
		if (!gCDevice) { result = -1; return; }
		gCQueue = [gCDevice newCommandQueue];
		if (!gCQueue) { result = -1; return; }
		NSString *path = [NSString stringWithUTF8String:metallib_path];
		NSError *err = nil;
		id<MTLLibrary> lib = [gCDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
		if (!lib) { result = -1; return; }

		gPSO_axpy = c_make_pso(gCDevice, lib, @"causal_axpy_kernel");
		gPSO_sub = c_make_pso(gCDevice, lib, @"causal_sub_kernel");
		gPSO_matvec = c_make_pso(gCDevice, lib, @"causal_matvec_kernel");
		gPSO_dotatom = c_make_pso(gCDevice, lib, @"causal_dot_atomic_kernel");
		gPSO_ata = c_make_pso(gCDevice, lib, @"causal_ata_kernel");
		gPSO_atb = c_make_pso(gCDevice, lib, @"causal_atb_kernel");
		gPSO_matmul = c_make_pso(gCDevice, lib, @"causal_matmul_kernel");
		gPSO_chol_inv = c_make_pso(gCDevice, lib, @"causal_chol_inv_kernel");

		gPSO_docalc_extract = c_make_pso(gCDevice, lib, @"docalc_extract_kernel");
		gPSO_docalc_assemble = c_make_pso(gCDevice, lib, @"docalc_assemble_kernel");

		gPSO_backdoor_design = c_make_pso(gCDevice, lib, @"backdoor_design_kernel");
		gPSO_backdoor_effect = c_make_pso(gCDevice, lib, @"backdoor_effect_kernel");

		gPSO_cate_split = c_make_pso(gCDevice, lib, @"cate_split_kernel");
		gPSO_cate_effect = c_make_pso(gCDevice, lib, @"cate_effect_kernel");

		gPSO_counterfactual = c_make_pso(gCDevice, lib, @"counterfactual_kernel");
		gPSO_frontdoor_sort_pad = c_make_pso(gCDevice, lib, @"frontdoor_sort_pad_kernel");
		gPSO_frontdoor_sort_step = c_make_pso(gCDevice, lib, @"frontdoor_sort_step_kernel");
		gPSO_frontdoor_boundaries = c_make_pso(gCDevice, lib, @"frontdoor_boundaries_kernel");
		gPSO_frontdoor_assign = c_make_pso(gCDevice, lib, @"frontdoor_assign_bins_kernel");
		gPSO_frontdoor_accumulate = c_make_pso(gCDevice, lib, @"frontdoor_accumulate_kernel");
		gPSO_frontdoor_normalize = c_make_pso(gCDevice, lib, @"frontdoor_normalize_kernel");
		gPSO_frontdoor_effect = c_make_pso(gCDevice, lib, @"frontdoor_effect_kernel");

		gPSO_dag_prep = c_make_pso(gCDevice, lib, @"dag_markov_prep_kernel");
		gPSO_dag_sigma2 = c_make_pso(gCDevice, lib, @"dag_markov_sigma2_kernel");
		gPSO_dag_score = c_make_pso(gCDevice, lib, @"dag_markov_score_kernel");

		if (!gPSO_axpy || !gPSO_sub || !gPSO_matvec || !gPSO_dotatom || !gPSO_ata || !gPSO_atb || !gPSO_matmul || !gPSO_chol_inv ||
		    !gPSO_docalc_extract || !gPSO_docalc_assemble || !gPSO_backdoor_design || !gPSO_backdoor_effect ||
		    !gPSO_cate_split || !gPSO_cate_effect || !gPSO_counterfactual ||
		    !gPSO_frontdoor_sort_pad || !gPSO_frontdoor_sort_step || !gPSO_frontdoor_boundaries ||
		    !gPSO_frontdoor_assign || !gPSO_frontdoor_accumulate || !gPSO_frontdoor_normalize ||
		    !gPSO_frontdoor_effect || !gPSO_dag_prep || !gPSO_dag_sigma2 || !gPSO_dag_score) {
			result = -1;
			return;
		}
		gCInited = 1;
	});
	return result;
}

int metal_causal_shutdown(void) {
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		gPSO_axpy = nil; gPSO_sub = nil; gPSO_matvec = nil; gPSO_dotatom = nil;
		gPSO_ata = nil; gPSO_atb = nil; gPSO_matmul = nil; gPSO_chol_inv = nil;
		gPSO_docalc_extract = nil; gPSO_docalc_assemble = nil;
		gPSO_backdoor_design = nil; gPSO_backdoor_effect = nil;
		gPSO_cate_split = nil; gPSO_cate_effect = nil;
		gPSO_counterfactual = nil; gPSO_frontdoor_boundaries = nil; gPSO_frontdoor_assign = nil;
		gPSO_frontdoor_sort_pad = nil; gPSO_frontdoor_sort_step = nil;
		gPSO_frontdoor_accumulate = nil; gPSO_frontdoor_normalize = nil; gPSO_frontdoor_effect = nil;
		gPSO_dag_prep = nil; gPSO_dag_sigma2 = nil; gPSO_dag_score = nil;
		gCQueue = nil; gCDevice = nil; gCInited = 0;
	});
	return 0;
}

static int c_check(void) { return gCInited ? 0 : -3; }

int metal_causal_axpy(float *dst, const float *src, float scale, int n) {
	if (c_check()) return -3;
	if (!dst || !src || n <= 0) return -1;
	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		size_t nb = (size_t)n * sizeof(float);
		id<MTLBuffer> buf_d = [gCDevice newBufferWithBytes:dst length:nb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_s = [gCDevice newBufferWithBytes:src length:nb options:MTLResourceStorageModeShared];
		if (!buf_d || !buf_s) { rc = -4; return; }
		id<MTLCommandBuffer> cb = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_axpy];
		[enc setBuffer:buf_d offset:0 atIndex:0];
		[enc setBuffer:buf_s offset:0 atIndex:1];
		[enc setBytes:&scale length:sizeof(scale) atIndex:2];
		[enc setBytes:&n length:sizeof(n) atIndex:3];
		[enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(gPSO_axpy.threadExecutionWidth, 1, 1)];
		[enc endEncoding];
		rc = c_wait(cb);
		if (rc == 0) memcpy(dst, [buf_d contents], nb);
	});
	return rc;
}

int metal_causal_dot(const float *a, const float *b, float *out, int n) {
	if (c_check()) return -3;
	if (!a || !b || !out || n <= 0) return -1;
	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		size_t nb = (size_t)n * sizeof(float);
		float zero = 0.f;
		id<MTLBuffer> buf_a = [gCDevice newBufferWithBytes:a length:nb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_b = [gCDevice newBufferWithBytes:b length:nb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_s = [gCDevice newBufferWithBytes:&zero length:sizeof(float) options:MTLResourceStorageModeShared];
		if (!buf_a || !buf_b || !buf_s) { rc = -4; return; }
		id<MTLCommandBuffer> cb = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_dotatom];
		[enc setBuffer:buf_a offset:0 atIndex:0];
		[enc setBuffer:buf_b offset:0 atIndex:1];
		[enc setBuffer:buf_s offset:0 atIndex:2];
		[enc setBytes:&n length:sizeof(n) atIndex:3];
		[enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(gPSO_dotatom.threadExecutionWidth, 1, 1)];
		[enc endEncoding];
		rc = c_wait(cb);
		if (rc == 0) memcpy(out, [buf_s contents], sizeof(float));
	});
	return rc;
}

int metal_causal_sub(float *dst, const float *a, const float *b, int n) {
	if (c_check()) return -3;
	if (!dst || !a || !b || n <= 0) return -1;
	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		size_t nb = (size_t)n * sizeof(float);
		id<MTLBuffer> buf_d = [gCDevice newBufferWithLength:nb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_a = [gCDevice newBufferWithBytes:a length:nb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_b = [gCDevice newBufferWithBytes:b length:nb options:MTLResourceStorageModeShared];
		if (!buf_d || !buf_a || !buf_b) { rc = -4; return; }
		id<MTLCommandBuffer> cb = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_sub];
		[enc setBuffer:buf_d offset:0 atIndex:0];
		[enc setBuffer:buf_a offset:0 atIndex:1];
		[enc setBuffer:buf_b offset:0 atIndex:2];
		[enc setBytes:&n length:sizeof(n) atIndex:3];
		[enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(gPSO_sub.threadExecutionWidth, 1, 1)];
		[enc endEncoding];
		rc = c_wait(cb);
		if (rc == 0) memcpy(dst, [buf_d contents], nb);
	});
	return rc;
}

int metal_causal_matvec(float *dst, const float *W, const float *x, int rows, int cols) {
	if (c_check()) return -3;
	if (!dst || !W || !x || rows <= 0 || cols <= 0) return -1;
	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		size_t wb = (size_t)rows * (size_t)cols * sizeof(float);
		size_t xb = (size_t)cols * sizeof(float);
		size_t db = (size_t)rows * sizeof(float);
		id<MTLBuffer> buf_d = [gCDevice newBufferWithLength:db options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_w = [gCDevice newBufferWithBytes:W length:wb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_x = [gCDevice newBufferWithBytes:x length:xb options:MTLResourceStorageModeShared];
		if (!buf_d || !buf_w || !buf_x) { rc = -4; return; }
		id<MTLCommandBuffer> cb = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_matvec];
		[enc setBuffer:buf_d offset:0 atIndex:0];
		[enc setBuffer:buf_w offset:0 atIndex:1];
		[enc setBuffer:buf_x offset:0 atIndex:2];
		[enc setBytes:&rows length:sizeof(rows) atIndex:3];
		[enc setBytes:&cols length:sizeof(cols) atIndex:4];
		[enc dispatchThreads:MTLSizeMake((NSUInteger)rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(gPSO_matvec.threadExecutionWidth, 1, 1)];
		[enc endEncoding];
		rc = c_wait(cb);
		if (rc == 0) memcpy(dst, [buf_d contents], db);
	});
	return rc;
}

// OLS Helper (dispatches ata, chol_inv, atb, matvec)
static void dispatch_ols(id<MTLCommandBuffer> cb, id<MTLBuffer> X, id<MTLBuffer> Y, id<MTLBuffer> beta, id<MTLBuffer> errBuf, int T, int p, int ny, float ridge, id<MTLBuffer> xtx, id<MTLBuffer> inv, id<MTLBuffer> xty, id<MTLBuffer> work) {
	id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
	[enc setComputePipelineState:gPSO_ata];
	[enc setBuffer:X offset:0 atIndex:0];
	[enc setBuffer:xtx offset:0 atIndex:1];
	[enc setBytes:&T length:sizeof(T) atIndex:2];
	[enc setBytes:&p length:sizeof(p) atIndex:3];
	[enc setBytes:&ridge length:sizeof(ridge) atIndex:4];
	[enc dispatchThreads:MTLSizeMake(p, p, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
	[enc endEncoding];

	enc = [cb computeCommandEncoder];
	[enc setComputePipelineState:gPSO_chol_inv];
	[enc setBuffer:xtx offset:0 atIndex:0];
	[enc setBuffer:inv offset:0 atIndex:1];
	[enc setBuffer:work offset:0 atIndex:2];
	[enc setBuffer:errBuf offset:0 atIndex:3];
	[enc setBytes:&p length:sizeof(p) atIndex:4];
	[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
	[enc endEncoding];

	enc = [cb computeCommandEncoder];
	[enc setComputePipelineState:gPSO_atb];
	[enc setBuffer:X offset:0 atIndex:0];
	[enc setBuffer:Y offset:0 atIndex:1];
	[enc setBuffer:xty offset:0 atIndex:2];
	[enc setBytes:&T length:sizeof(T) atIndex:3];
	[enc setBytes:&p length:sizeof(p) atIndex:4];
	[enc setBytes:&ny length:sizeof(ny) atIndex:5];
	[enc dispatchThreads:MTLSizeMake(p, ny, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
	[enc endEncoding];

	enc = [cb computeCommandEncoder];
	[enc setComputePipelineState:gPSO_matmul];
	[enc setBuffer:inv offset:0 atIndex:0];
	[enc setBuffer:xty offset:0 atIndex:1];
	[enc setBuffer:beta offset:0 atIndex:2];
	[enc setBytes:&p length:sizeof(p) atIndex:3];
	[enc setBytes:&p length:sizeof(p) atIndex:4];
	[enc setBytes:&ny length:sizeof(ny) atIndex:5];
	[enc dispatchThreads:MTLSizeMake(p, ny, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
	[enc endEncoding];
}

int metal_causal_do_calculus(const float *cov_f, const float *mask_f, const float *values_f, float *out_f, int N) {
	if (c_check()) return -3;
	if (!cov_f || !mask_f || !values_f || !out_f || N <= 0) return -1;
	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		id<MTLBuffer> b_cov = [gCDevice newBufferWithBytes:cov_f length:N*N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_mask = [gCDevice newBufferWithBytes:mask_f length:N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_val = [gCDevice newBufferWithBytes:values_f length:N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_out = [gCDevice newBufferWithLength:(N+N*N)*sizeof(float) options:MTLResourceStorageModeShared];
		
		id<MTLBuffer> sigII = [gCDevice newBufferWithLength:N*N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> sigFI = [gCDevice newBufferWithLength:N*N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> sigFF = [gCDevice newBufferWithLength:N*N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> sigIF = [gCDevice newBufferWithLength:N*N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> xIntV = [gCDevice newBufferWithLength:N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> intervened = [gCDevice newBufferWithLength:N*sizeof(int) options:MTLResourceStorageModeShared];
		id<MTLBuffer> freev = [gCDevice newBufferWithLength:N*sizeof(int) options:MTLResourceStorageModeShared];
		id<MTLBuffer> counts = [gCDevice newBufferWithLength:2*sizeof(int) options:MTLResourceStorageModeShared];
		
		id<MTLBuffer> invII = [gCDevice newBufferWithLength:N*N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> invX = [gCDevice newBufferWithLength:N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> delta = [gCDevice newBufferWithLength:N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> invIISigIF = [gCDevice newBufferWithLength:N*N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> correction = [gCDevice newBufferWithLength:N*N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> work = [gCDevice newBufferWithLength:(N*N+2*N)*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> errBuf = [gCDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];

		id<MTLCommandBuffer> cb = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_docalc_extract];
		[enc setBuffer:b_cov offset:0 atIndex:0]; [enc setBuffer:b_mask offset:0 atIndex:1]; [enc setBuffer:b_val offset:0 atIndex:2];
		[enc setBuffer:sigII offset:0 atIndex:3]; [enc setBuffer:sigFI offset:0 atIndex:4]; [enc setBuffer:sigFF offset:0 atIndex:5];
		[enc setBuffer:sigIF offset:0 atIndex:6]; [enc setBuffer:xIntV offset:0 atIndex:7]; [enc setBuffer:intervened offset:0 atIndex:8];
		[enc setBuffer:freev offset:0 atIndex:9]; [enc setBuffer:counts offset:0 atIndex:10]; [enc setBytes:&N length:sizeof(N) atIndex:11];
		[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc endEncoding];
		rc = c_wait(cb);
		if (rc != 0) return;

		int ni = ((int*)[counts contents])[0];
		int nf = ((int*)[counts contents])[1];

		if (ni > 0 && nf > 0) {
			cb = [gCQueue commandBuffer];
			enc = [cb computeCommandEncoder];
			[enc setComputePipelineState:gPSO_chol_inv];
			[enc setBuffer:sigII offset:0 atIndex:0]; [enc setBuffer:invII offset:0 atIndex:1]; [enc setBuffer:work offset:0 atIndex:2];
			[enc setBuffer:errBuf offset:0 atIndex:3]; [enc setBytes:&ni length:sizeof(ni) atIndex:4];
			[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
			[enc endEncoding];

			enc = [cb computeCommandEncoder];
			[enc setComputePipelineState:gPSO_matvec];
			[enc setBuffer:invX offset:0 atIndex:0]; [enc setBuffer:invII offset:0 atIndex:1]; [enc setBuffer:xIntV offset:0 atIndex:2];
			[enc setBytes:&ni length:sizeof(ni) atIndex:3]; [enc setBytes:&ni length:sizeof(ni) atIndex:4];
			[enc dispatchThreads:MTLSizeMake(ni, 1, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
			[enc endEncoding];

			enc = [cb computeCommandEncoder];
			[enc setComputePipelineState:gPSO_matvec];
			[enc setBuffer:delta offset:0 atIndex:0]; [enc setBuffer:sigFI offset:0 atIndex:1]; [enc setBuffer:invX offset:0 atIndex:2];
			[enc setBytes:&nf length:sizeof(nf) atIndex:3]; [enc setBytes:&ni length:sizeof(ni) atIndex:4];
			[enc dispatchThreads:MTLSizeMake(nf, 1, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
			[enc endEncoding];

			enc = [cb computeCommandEncoder];
			[enc setComputePipelineState:gPSO_matmul];
			[enc setBuffer:invII offset:0 atIndex:0]; [enc setBuffer:sigIF offset:0 atIndex:1]; [enc setBuffer:invIISigIF offset:0 atIndex:2];
			[enc setBytes:&ni length:sizeof(ni) atIndex:3]; [enc setBytes:&ni length:sizeof(ni) atIndex:4]; [enc setBytes:&nf length:sizeof(nf) atIndex:5];
			[enc dispatchThreads:MTLSizeMake(ni, nf, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
			[enc endEncoding];

			enc = [cb computeCommandEncoder];
			[enc setComputePipelineState:gPSO_matmul];
			[enc setBuffer:sigFI offset:0 atIndex:0]; [enc setBuffer:invIISigIF offset:0 atIndex:1]; [enc setBuffer:correction offset:0 atIndex:2];
			[enc setBytes:&nf length:sizeof(nf) atIndex:3]; [enc setBytes:&ni length:sizeof(ni) atIndex:4]; [enc setBytes:&nf length:sizeof(nf) atIndex:5];
			[enc dispatchThreads:MTLSizeMake(nf, nf, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
			[enc endEncoding];
			rc = c_wait(cb);
			if (rc != 0) return;
			if (((int*)[errBuf contents])[0] != 0) { rc = -5; return; }
		}

		cb = [gCQueue commandBuffer];
		enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_docalc_assemble];
		[enc setBuffer:b_out offset:0 atIndex:0]; [enc setBuffer:b_val offset:0 atIndex:1]; [enc setBuffer:delta offset:0 atIndex:2];
		[enc setBuffer:sigFF offset:0 atIndex:3]; [enc setBuffer:correction offset:0 atIndex:4]; [enc setBuffer:intervened offset:0 atIndex:5];
		[enc setBuffer:freev offset:0 atIndex:6]; [enc setBuffer:counts offset:0 atIndex:7]; [enc setBytes:&N length:sizeof(N) atIndex:8];
		[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc endEncoding];
		rc = c_wait(cb);
		if (rc == 0) memcpy(out_f, [b_out contents], (N+N*N)*sizeof(float));
	});
	return rc;
}

int metal_causal_backdoor(const float *Y, const float *X, const float *Z, float *effect, int T, int ny, int nx, int nz) {
	if (c_check()) return -3;
	if (!Y || !X || !Z || !effect || T <= 0 || ny <= 0 || nx <= 0 || nz < 0) return -1;
	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		int p = 1 + nx + nz;
		id<MTLBuffer> b_Y = [gCDevice newBufferWithBytes:Y length:T*ny*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_X = [gCDevice newBufferWithBytes:X length:T*nx*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_Z = [gCDevice newBufferWithBytes:Z length:T*nz*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_eff = [gCDevice newBufferWithLength:ny*sizeof(float) options:MTLResourceStorageModeShared];
		
		id<MTLBuffer> design = [gCDevice newBufferWithLength:T*p*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> xtx = [gCDevice newBufferWithLength:p*p*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> inv = [gCDevice newBufferWithLength:p*p*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> xty = [gCDevice newBufferWithLength:p*ny*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> beta = [gCDevice newBufferWithLength:p*ny*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> work = [gCDevice newBufferWithLength:(p*p+2*p)*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> errBuf = [gCDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];

		id<MTLCommandBuffer> cb = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_backdoor_design];
		[enc setBuffer:b_X offset:0 atIndex:0]; [enc setBuffer:b_Z offset:0 atIndex:1]; [enc setBuffer:design offset:0 atIndex:2];
		[enc setBytes:&T length:sizeof(T) atIndex:3]; [enc setBytes:&nx length:sizeof(nx) atIndex:4]; [enc setBytes:&nz length:sizeof(nz) atIndex:5];
		[enc dispatchThreads:MTLSizeMake(T, p, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
		[enc endEncoding];
		
		dispatch_ols(cb, design, b_Y, beta, errBuf, T, p, ny, 1e-10f, xtx, inv, xty, work);
		
		enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_backdoor_effect];
		[enc setBuffer:beta offset:0 atIndex:0]; [enc setBuffer:b_eff offset:0 atIndex:1];
		[enc setBytes:&ny length:sizeof(ny) atIndex:2]; [enc setBytes:&nx length:sizeof(nx) atIndex:3]; [enc setBytes:&p length:sizeof(p) atIndex:4];
		[enc dispatchThreads:MTLSizeMake(ny, 1, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
		[enc endEncoding];
		
		rc = c_wait(cb);
		if (rc != 0) return;
		if (((int*)[errBuf contents])[0] != 0) { rc = -5; return; }
		memcpy(effect, [b_eff contents], ny*sizeof(float));
	});
	return rc;
}

int metal_causal_iv(const float *Z, const float *X, const float *Y, float *beta_iv, int T, int nz, int nx, int ny) {
	if (c_check()) return -3;
	if (!Z || !X || !Y || !beta_iv || T <= 0 || nz <= 0 || nx <= 0 || ny <= 0) return -1;
	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		id<MTLBuffer> b_Z = [gCDevice newBufferWithBytes:Z length:T*nz*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_X = [gCDevice newBufferWithBytes:X length:T*nx*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_Y = [gCDevice newBufferWithBytes:Y length:T*ny*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_beta = [gCDevice newBufferWithLength:nx*ny*sizeof(float) options:MTLResourceStorageModeShared];
		
		id<MTLBuffer> ztz = [gCDevice newBufferWithLength:nz*nz*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> ztzInv = [gCDevice newBufferWithLength:nz*nz*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> ztx = [gCDevice newBufferWithLength:nz*nx*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> proj = [gCDevice newBufferWithLength:nz*nx*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> xHat = [gCDevice newBufferWithLength:T*nx*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> xtx = [gCDevice newBufferWithLength:nx*nx*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> inv = [gCDevice newBufferWithLength:nx*nx*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> xty = [gCDevice newBufferWithLength:nx*ny*sizeof(float) options:MTLResourceStorageModeShared];
		
		int maxN = nz > nx ? nz : nx;
		id<MTLBuffer> work = [gCDevice newBufferWithLength:(maxN*maxN+2*maxN)*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> errBuf = [gCDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];

		id<MTLCommandBuffer> cb = [gCQueue commandBuffer];
		dispatch_ols(cb, b_Z, b_X, proj, errBuf, T, nz, nx, 1e-10f, ztz, ztzInv, ztx, work);
		
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_matmul];
		[enc setBuffer:b_Z offset:0 atIndex:0]; [enc setBuffer:proj offset:0 atIndex:1]; [enc setBuffer:xHat offset:0 atIndex:2];
		[enc setBytes:&T length:sizeof(T) atIndex:3]; [enc setBytes:&nz length:sizeof(nz) atIndex:4]; [enc setBytes:&nx length:sizeof(nx) atIndex:5];
		[enc dispatchThreads:MTLSizeMake(T, nx, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
		[enc endEncoding];
		
		dispatch_ols(cb, xHat, b_Y, b_beta, errBuf, T, nx, ny, 1e-10f, xtx, inv, xty, work);
		
		rc = c_wait(cb);
		if (rc != 0) return;
		if (((int*)[errBuf contents])[0] != 0) { rc = -5; return; }
		memcpy(beta_iv, [b_beta contents], nx*ny*sizeof(float));
	});
	return rc;
}

int metal_causal_cate(const float *X, const float *treatment, const float *Y, float *cate, int T, int nx) {
	if (c_check()) return -3;
	if (!X || !treatment || !Y || !cate || T <= 0 || nx <= 0) return -1;
	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		int nFeat = nx + 1;
		id<MTLBuffer> b_X = [gCDevice newBufferWithBytes:X length:T*nx*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_T = [gCDevice newBufferWithBytes:treatment length:T*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_Y = [gCDevice newBufferWithBytes:Y length:T*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_cate = [gCDevice newBufferWithLength:T*sizeof(float) options:MTLResourceStorageModeShared];
		
		id<MTLBuffer> xSub1 = [gCDevice newBufferWithLength:T*nFeat*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> ySub1 = [gCDevice newBufferWithLength:T*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> xSub0 = [gCDevice newBufferWithLength:T*nFeat*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> ySub0 = [gCDevice newBufferWithLength:T*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> counts = [gCDevice newBufferWithLength:2*sizeof(int) options:MTLResourceStorageModeShared];
		
		id<MTLBuffer> xtx = [gCDevice newBufferWithLength:nFeat*nFeat*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> inv = [gCDevice newBufferWithLength:nFeat*nFeat*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> xty = [gCDevice newBufferWithLength:nFeat*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b1 = [gCDevice newBufferWithLength:nFeat*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b0 = [gCDevice newBufferWithLength:nFeat*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> work = [gCDevice newBufferWithLength:(nFeat*nFeat+2*nFeat)*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> errBuf = [gCDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];

		id<MTLCommandBuffer> cb = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_cate_split];
		[enc setBuffer:b_X offset:0 atIndex:0]; [enc setBuffer:b_T offset:0 atIndex:1]; [enc setBuffer:b_Y offset:0 atIndex:2];
		[enc setBuffer:xSub1 offset:0 atIndex:3]; [enc setBuffer:ySub1 offset:0 atIndex:4];
		[enc setBuffer:xSub0 offset:0 atIndex:5]; [enc setBuffer:ySub0 offset:0 atIndex:6];
		[enc setBuffer:counts offset:0 atIndex:7]; [enc setBytes:&T length:sizeof(T) atIndex:8]; [enc setBytes:&nx length:sizeof(nx) atIndex:9];
		[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc endEncoding];
		rc = c_wait(cb);
		if (rc != 0) return;
		
		int nt = ((int*)[counts contents])[0];
		int nc = ((int*)[counts contents])[1];
		if (nt == 0 || nc == 0) {
			for (int i=0; i<T; i++) cate[i] = NAN;
			return;
		}

		cb = [gCQueue commandBuffer];
		dispatch_ols(cb, xSub1, ySub1, b1, errBuf, nt, nFeat, 1, 1e-10f, xtx, inv, xty, work);
		dispatch_ols(cb, xSub0, ySub0, b0, errBuf, nc, nFeat, 1, 1e-10f, xtx, inv, xty, work);
		
		enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_cate_effect];
		[enc setBuffer:b_X offset:0 atIndex:0]; [enc setBuffer:b1 offset:0 atIndex:1]; [enc setBuffer:b0 offset:0 atIndex:2];
		[enc setBuffer:b_cate offset:0 atIndex:3]; [enc setBytes:&T length:sizeof(T) atIndex:4]; [enc setBytes:&nx length:sizeof(nx) atIndex:5];
		[enc dispatchThreads:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
		[enc endEncoding];
		
		rc = c_wait(cb);
		if (rc != 0) return;
		if (((int*)[errBuf contents])[0] != 0) { rc = -5; return; }
		memcpy(cate, [b_cate contents], T*sizeof(float));
	});
	return rc;
}

int metal_causal_counterfactual(
	const float *X_obs, const float *Y_obs, const float *beta, const float *X_cf,
	float *out, int N, int N_cf)
{
	if (c_check()) return -3;
	if (!X_obs || !Y_obs || !beta || !X_cf || !out || N <= 0 || N_cf <= 0) return -1;
	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		size_t obsBytes = (size_t)N * sizeof(float);
		size_t cfBytes = (size_t)N_cf * sizeof(float);
		size_t outBytes = (size_t)N * (size_t)N_cf * sizeof(float);
		id<MTLBuffer> bX = [gCDevice newBufferWithBytes:X_obs length:obsBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> bY = [gCDevice newBufferWithBytes:Y_obs length:obsBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> bBeta = [gCDevice newBufferWithBytes:beta length:obsBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> bXcf = [gCDevice newBufferWithBytes:X_cf length:cfBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> bOut = [gCDevice newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
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
		if (rc == 0) memcpy(out, [bOut contents], outBytes);
	});
	return rc;
}

int metal_causal_frontdoor(
	const float *X, const float *M, const float *Y,
	float *effect, int T, int nx, int nm)
{
	if (c_check()) return -3;
	if (!X || !M || !Y || !effect || T <= 0 || nx <= 0 || nm <= 0) return -1;
	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		size_t sampleBytes = (size_t)T * sizeof(float);
		size_t xBoundaryBytes = (size_t)(nx > 1 ? nx - 1 : 1) * sizeof(float);
		size_t mBoundaryBytes = (size_t)(nm > 1 ? nm - 1 : 1) * sizeof(float);
		size_t xBytes = (size_t)nx * sizeof(float);
		size_t matrixBytes = (size_t)nx * (size_t)nm * sizeof(float);
		size_t binBytes = (size_t)T * sizeof(int);
		int paddedSamples = next_power_of_two_int(T);
		size_t sortedBytes = (size_t)paddedSamples * sizeof(float);

		id<MTLBuffer> bX = [gCDevice newBufferWithBytes:X length:sampleBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> bM = [gCDevice newBufferWithBytes:M length:sampleBytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> bY = [gCDevice newBufferWithBytes:Y length:sampleBytes options:MTLResourceStorageModeShared];
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
		id<MTLBuffer> bEffect = [gCDevice newBufferWithLength:xBytes options:MTLResourceStorageModeShared];

		if (!bX || !bM || !bY || !bXBoundaries || !bMBoundaries || !bXBins || !bMBins ||
		    !bPX || !bCountX || !bMGivenX || !bEYGivenXM || !bCountXM || !bEffect) {
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

		if (nx > 1) {
			encode_frontdoor_sort(cb, bX, bXSorted, T, paddedSamples);
		}

		if (nm > 1) {
			encode_frontdoor_sort(cb, bM, bMSorted, T, paddedSamples);
		}

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
		[encX setBuffer:bX offset:0 atIndex:0];
		[encX setBuffer:bXBoundaries offset:0 atIndex:1];
		[encX setBuffer:bXBins offset:0 atIndex:2];
		[encX setBytes:&T length:sizeof(T) atIndex:3];
		[encX setBytes:&nx length:sizeof(nx) atIndex:4];
		[encX dispatchThreads:MTLSizeMake((NSUInteger)T, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(gPSO_frontdoor_assign.threadExecutionWidth, 1, 1)];
		[encX endEncoding];

		id<MTLComputeCommandEncoder> encM = [cb computeCommandEncoder];
		[encM setComputePipelineState:gPSO_frontdoor_assign];
		[encM setBuffer:bM offset:0 atIndex:0];
		[encM setBuffer:bMBoundaries offset:0 atIndex:1];
		[encM setBuffer:bMBins offset:0 atIndex:2];
		[encM setBytes:&T length:sizeof(T) atIndex:3];
		[encM setBytes:&nm length:sizeof(nm) atIndex:4];
		[encM dispatchThreads:MTLSizeMake((NSUInteger)T, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(gPSO_frontdoor_assign.threadExecutionWidth, 1, 1)];
		[encM endEncoding];

		id<MTLComputeCommandEncoder> encA = [cb computeCommandEncoder];
		[encA setComputePipelineState:gPSO_frontdoor_accumulate];
		[encA setBuffer:bXBins offset:0 atIndex:0];
		[encA setBuffer:bMBins offset:0 atIndex:1];
		[encA setBuffer:bY offset:0 atIndex:2];
		[encA setBuffer:bPX offset:0 atIndex:3];
		[encA setBuffer:bCountX offset:0 atIndex:4];
		[encA setBuffer:bMGivenX offset:0 atIndex:5];
		[encA setBuffer:bEYGivenXM offset:0 atIndex:6];
		[encA setBuffer:bCountXM offset:0 atIndex:7];
		[encA setBytes:&T length:sizeof(T) atIndex:8];
		[encA setBytes:&nx length:sizeof(nx) atIndex:9];
		[encA setBytes:&nm length:sizeof(nm) atIndex:10];
		[encA dispatchThreads:MTLSizeMake((NSUInteger)T, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(gPSO_frontdoor_accumulate.threadExecutionWidth, 1, 1)];
		[encA endEncoding];

		id<MTLComputeCommandEncoder> encN = [cb computeCommandEncoder];
		[encN setComputePipelineState:gPSO_frontdoor_normalize];
		[encN setBuffer:bPX offset:0 atIndex:0];
		[encN setBuffer:bCountX offset:0 atIndex:1];
		[encN setBuffer:bMGivenX offset:0 atIndex:2];
		[encN setBuffer:bEYGivenXM offset:0 atIndex:3];
		[encN setBuffer:bCountXM offset:0 atIndex:4];
		[encN setBytes:&T length:sizeof(T) atIndex:5];
		[encN setBytes:&nx length:sizeof(nx) atIndex:6];
		[encN setBytes:&nm length:sizeof(nm) atIndex:7];
		NSUInteger normalizeThreads = (NSUInteger)nx * (NSUInteger)nm;
		[encN dispatchThreads:MTLSizeMake(normalizeThreads, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(gPSO_frontdoor_normalize.threadExecutionWidth, 1, 1)];
		[encN endEncoding];

		id<MTLComputeCommandEncoder> encE = [cb computeCommandEncoder];
		[encE setComputePipelineState:gPSO_frontdoor_effect];
		[encE setBuffer:bPX offset:0 atIndex:0];
		[encE setBuffer:bMGivenX offset:0 atIndex:1];
		[encE setBuffer:bEYGivenXM offset:0 atIndex:2];
		[encE setBuffer:bEffect offset:0 atIndex:3];
		[encE setBytes:&nx length:sizeof(nx) atIndex:4];
		[encE setBytes:&nm length:sizeof(nm) atIndex:5];
		[encE dispatchThreads:MTLSizeMake((NSUInteger)nx, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(gPSO_frontdoor_effect.threadExecutionWidth, 1, 1)];
		[encE endEncoding];

		rc = c_wait(cb);
		if (rc == 0) memcpy(effect, [bEffect contents], xBytes);
	});
	return rc;
}

int metal_causal_dag_markov(const float *X, const float *adj, float *log_prob, int T, int N) {
	if (c_check()) return -3;
	if (!X || !adj || !log_prob || T <= 0 || N <= 0) return -1;
	__block int rc = 0;
	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		id<MTLBuffer> b_X = [gCDevice newBufferWithBytes:X length:T*N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_adj = [gCDevice newBufferWithBytes:adj length:N*N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_log = [gCDevice newBufferWithLength:T*sizeof(float) options:MTLResourceStorageModeShared];
		
		id<MTLBuffer> betas = [gCDevice newBufferWithLength:N*(N+1)*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> sigma2 = [gCDevice newBufferWithLength:N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> pMat = [gCDevice newBufferWithLength:T*(N+1)*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> nodeVals = [gCDevice newBufferWithLength:T*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> counts = [gCDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
		
		id<MTLBuffer> xtx = [gCDevice newBufferWithLength:(N+1)*(N+1)*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> inv = [gCDevice newBufferWithLength:(N+1)*(N+1)*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> xty = [gCDevice newBufferWithLength:(N+1)*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> work = [gCDevice newBufferWithLength:((N+1)*(N+1)+2*(N+1))*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> errBuf = [gCDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];

		for (int nodeIdx = 0; nodeIdx < N; nodeIdx++) {
			id<MTLCommandBuffer> cb = [gCQueue commandBuffer];
			id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
			[enc setComputePipelineState:gPSO_dag_prep];
			[enc setBuffer:b_X offset:0 atIndex:0]; [enc setBuffer:b_adj offset:0 atIndex:1]; [enc setBuffer:pMat offset:0 atIndex:2];
			[enc setBuffer:nodeVals offset:0 atIndex:3]; [enc setBuffer:counts offset:0 atIndex:4];
			[enc setBytes:&T length:sizeof(T) atIndex:5]; [enc setBytes:&N length:sizeof(N) atIndex:6]; [enc setBytes:&nodeIdx length:sizeof(nodeIdx) atIndex:7];
			[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
			[enc endEncoding];
			rc = c_wait(cb);
			if (rc != 0) return;
			
			int np = ((int*)[counts contents])[0];
			int nFeat = np + 1;
			id<MTLBuffer> bDest = [gCDevice newBufferWithBytesNoCopy:((float*)[betas contents]) + nodeIdx*(N+1) length:nFeat*sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
			
			cb = [gCQueue commandBuffer];
			dispatch_ols(cb, pMat, nodeVals, bDest, errBuf, T, nFeat, 1, 1e-10f, xtx, inv, xty, work);
			rc = c_wait(cb);
			if (rc != 0) return;
			if (((int*)[errBuf contents])[0] != 0) { rc = -5; return; }

			cb = [gCQueue commandBuffer];
			enc = [cb computeCommandEncoder];
			[enc setComputePipelineState:gPSO_dag_sigma2];
			[enc setBuffer:pMat offset:0 atIndex:0];
			[enc setBuffer:nodeVals offset:0 atIndex:1];
			[enc setBuffer:bDest offset:0 atIndex:2];
			[enc setBuffer:sigma2 offset:0 atIndex:3];
			[enc setBytes:&T length:sizeof(T) atIndex:4];
			[enc setBytes:&nFeat length:sizeof(nFeat) atIndex:5];
			[enc setBytes:&nodeIdx length:sizeof(nodeIdx) atIndex:6];
			[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
			[enc endEncoding];
			rc = c_wait(cb);
			if (rc != 0) return;
		}

		id<MTLCommandBuffer> cb = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_dag_score];
		[enc setBuffer:b_X offset:0 atIndex:0]; [enc setBuffer:b_adj offset:0 atIndex:1]; [enc setBuffer:betas offset:0 atIndex:2];
		[enc setBuffer:sigma2 offset:0 atIndex:3]; [enc setBuffer:b_log offset:0 atIndex:4];
		[enc setBytes:&T length:sizeof(T) atIndex:5]; [enc setBytes:&N length:sizeof(N) atIndex:6];
		[enc dispatchThreads:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
		[enc endEncoding];
		rc = c_wait(cb);
		if (rc == 0) memcpy(log_prob, [b_log contents], T*sizeof(float));
	});
	return rc;
}
