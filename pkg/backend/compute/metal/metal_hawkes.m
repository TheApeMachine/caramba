#include <stdint.h>
#include <limits.h>
#include <string.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "hawkes.h"

static id<MTLDevice>                gHDevice  = nil;
static id<MTLCommandQueue>          gHQueue   = nil;
static id<MTLComputePipelineState> gPSO_int  = nil;
static id<MTLComputePipelineState> gPSO_kmat  = nil;
static id<MTLComputePipelineState> gPSO_logt  = nil;
static id<MTLComputePipelineState> gPSO_logred = nil;
static id<MTLComputePipelineState> gPSO_logfin = nil;
static id<MTLComputePipelineState> gPSO_simclr = nil;
static id<MTLComputePipelineState> gPSO_simdim = nil;
static int                          gHInited  = 0;
static dispatch_queue_t             gHSerial  = NULL;

static void hawkes_ensure_serial(void) {
	static dispatch_once_t onceToken;

	dispatch_once(&onceToken, ^{
		gHSerial = dispatch_queue_create("com.caramba.metal.hawkes", DISPATCH_QUEUE_SERIAL);
	});
}

static id<MTLComputePipelineState> hawkes_make_pso(
	id<MTLDevice> device, id<MTLLibrary> library, NSString *name
) {
	NSError *err = nil;
	id<MTLFunction> fn = [library newFunctionWithName:name];

	if (!fn) {
		return nil;
	}

	return [device newComputePipelineStateWithFunction:fn error:&err];
}

static int hawkes_wait(id<MTLCommandBuffer> cb) {
	dispatch_semaphore_t done = dispatch_semaphore_create(0);

	if (!done) {
		return -1;
	}

	[cb addCompletedHandler:^(id<MTLCommandBuffer> _) {
		dispatch_semaphore_signal(done);
	}];
	[cb commit];
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);

	if (cb.error != nil) {
		return -3;
	}

	return 0;
}

int metal_hawkes_init(const char *metallib_path) {
	if (metallib_path == NULL || metallib_path[0] == '\0') {
		return -1;
	}

	hawkes_ensure_serial();
	__block int result = 0;

	dispatch_sync(gHSerial, ^{
		if (gHInited) {
			return;
		}

		gHDevice = MTLCreateSystemDefaultDevice();

		if (!gHDevice) {
			result = -1;
			return;
		}

		gHQueue = [gHDevice newCommandQueue];

		if (!gHQueue) {
			result = -1;
			return;
		}

		NSString *path = [NSString stringWithUTF8String:metallib_path];
		NSError *err   = nil;
		id<MTLLibrary> lib = [gHDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];

		if (!lib) {
			result = -1;
			return;
		}

		gPSO_int = hawkes_make_pso(gHDevice, lib, @"hawkes_intensity_kernel");
		gPSO_kmat = hawkes_make_pso(gHDevice, lib, @"hawkes_kernel_matrix_kernel");
		gPSO_logt = hawkes_make_pso(gHDevice, lib, @"hawkes_log_term_kernel");
		gPSO_logred = hawkes_make_pso(gHDevice, lib, @"hawkes_reduce_sum_atomic_kernel");
		gPSO_logfin = hawkes_make_pso(gHDevice, lib, @"hawkes_loglik_finalize_kernel");
		gPSO_simclr = hawkes_make_pso(gHDevice, lib, @"hawkes_sim_clear_kernel");
		gPSO_simdim = hawkes_make_pso(gHDevice, lib, @"hawkes_simulate_dim_kernel");

		if (
			!gPSO_int
			|| !gPSO_kmat
			|| !gPSO_logt
			|| !gPSO_logred
			|| !gPSO_logfin
			|| !gPSO_simclr
			|| !gPSO_simdim
		) {
			result = -1;
			return;
		}

		gHInited = 1;
	});

	return result;
}

int metal_hawkes_cleanup(void) {
	hawkes_ensure_serial();
	dispatch_sync(gHSerial, ^{
		gPSO_int  = nil;
		gPSO_kmat = nil;
		gPSO_logt = nil;
		gPSO_logred = nil;
		gPSO_logfin = nil;
		gPSO_simclr = nil;
		gPSO_simdim = nil;
		gHQueue   = nil;
		gHDevice  = nil;
		gHInited  = 0;
	});

	return 0;
}

int metal_hawkes_intensity(
	const float *times, const float *alpha,
	const float *beta, const float *mu,
	float t,
	float *out,
	int K, int T) {
	if (!gHInited) {
		return -3;
	}

	if (!times || !alpha || !beta || !mu || !out || K <= 0 || T < 0) {
		return -1;
	}

	__block int rc = 0;

	hawkes_ensure_serial();
	dispatch_sync(gHSerial, ^{
		size_t tbytes = (size_t)T * sizeof(float);
		size_t kbytes = (size_t)K * sizeof(float);
		id<MTLBuffer> buf_times = [gHDevice newBufferWithBytes:times
		                                                  length:tbytes
		                                                 options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_alpha = [gHDevice newBufferWithBytes:alpha
		                                                  length:kbytes
		                                                 options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_beta  = [gHDevice newBufferWithBytes:beta
		                                                  length:kbytes
		                                                 options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_mu    = [gHDevice newBufferWithBytes:mu
		                                               length:kbytes
		                                              options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_out   = [gHDevice newBufferWithLength:kbytes
		                                                options:MTLResourceStorageModeShared];

		if (!buf_times || !buf_alpha || !buf_beta || !buf_mu || !buf_out) {
			rc = -1;
			return;
		}

		id<MTLCommandBuffer>         cb = [gHQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_int];
		[enc setBuffer:buf_times offset:0 atIndex:0];
		[enc setBuffer:buf_alpha offset:0 atIndex:1];
		[enc setBuffer:buf_beta  offset:0 atIndex:2];
		[enc setBuffer:buf_mu    offset:0 atIndex:3];
		[enc setBuffer:buf_out   offset:0 atIndex:4];
		[enc setBytes:&t length:sizeof(t) atIndex:5];
		uint ku = (uint)K;
		uint tu = (uint)T;
		[enc setBytes:&ku length:sizeof(ku) atIndex:6];
		[enc setBytes:&tu length:sizeof(tu) atIndex:7];

		NSUInteger tw = gPSO_int.threadExecutionWidth;

		[enc dispatchThreads:MTLSizeMake((NSUInteger)K, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
		[enc endEncoding];

		rc = hawkes_wait(cb);

		if (rc != 0) {
			return;
		}

		memcpy(out, [buf_out contents], kbytes);
	});

	return rc;
}

int metal_hawkes_intensity_tensor(
	const void *times, const void *alpha,
	const void *beta, const void *mu,
	const void *t,
	void *out,
	int K, int T) {
	if (!gHInited) {
		return -3;
	}

	if (!times || !alpha || !beta || !mu || !t || !out || K <= 0 || T < 0) {
		return -1;
	}

	__block int rc = 0;

	hawkes_ensure_serial();
	dispatch_sync(gHSerial, ^{
		id<MTLBuffer> buf_times = (__bridge id)((void*)times);
		id<MTLBuffer> buf_alpha = (__bridge id)((void*)alpha);
		id<MTLBuffer> buf_beta = (__bridge id)((void*)beta);
		id<MTLBuffer> buf_mu = (__bridge id)((void*)mu);
		id<MTLBuffer> buf_t = (__bridge id)((void*)t);
		id<MTLBuffer> buf_out = (__bridge id)out;

		if (!buf_times || !buf_alpha || !buf_beta || !buf_mu || !buf_t || !buf_out) {
			rc = -1;
			return;
		}

		id<MTLCommandBuffer> cb = [gHQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		if (!cb || !enc) {
			rc = -1;
			return;
		}

		[enc setComputePipelineState:gPSO_int];
		[enc setBuffer:buf_times offset:0 atIndex:0];
		[enc setBuffer:buf_alpha offset:0 atIndex:1];
		[enc setBuffer:buf_beta offset:0 atIndex:2];
		[enc setBuffer:buf_mu offset:0 atIndex:3];
		[enc setBuffer:buf_out offset:0 atIndex:4];
		[enc setBuffer:buf_t offset:0 atIndex:5];
		uint ku = (uint)K;
		uint tu = (uint)T;
		[enc setBytes:&ku length:sizeof(ku) atIndex:6];
		[enc setBytes:&tu length:sizeof(tu) atIndex:7];

		[enc dispatchThreads:MTLSizeMake((NSUInteger)K, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(gPSO_int.threadExecutionWidth, 1, 1)];
		[enc endEncoding];

		rc = hawkes_wait(cb);
	});

	return rc;
}

int metal_hawkes_kernel_matrix(
	const float *times,
	float alpha, float beta,
	float *out,
	int T) {
	if (!gHInited) {
		return -3;
	}

	if (!times || !out || T <= 0) {
		return -1;
	}

	__block int rc = 0;

	hawkes_ensure_serial();
	dispatch_sync(gHSerial, ^{
		size_t         tb = (size_t)T * sizeof(float);
		size_t         ob = (size_t)T * (size_t)T * sizeof(float);
		id<MTLBuffer> buf_t = [gHDevice newBufferWithBytes:times
		                                             length:tb
		                                            options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_o = [gHDevice newBufferWithLength:ob
		                                            options:MTLResourceStorageModeShared];

		if (!buf_t || !buf_o) {
			rc = -1;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gHQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_kmat];
		[enc setBuffer:buf_t offset:0 atIndex:0];
		[enc setBuffer:buf_o offset:0 atIndex:1];
		uint tu = (uint)T;
		[enc setBytes:&tu length:sizeof(tu) atIndex:2];
		[enc setBytes:&alpha length:sizeof(alpha) atIndex:3];
		[enc setBytes:&beta length:sizeof(beta) atIndex:4];

		NSUInteger tx = 16;
		NSUInteger ty = 16;

		if (T < 16) {
			tx = (NSUInteger)T;
		}

		if (T < 16) {
			ty = (NSUInteger)T;
		}

		if (tx < 1) {
			tx = 1;
		}

		if (ty < 1) {
			ty = 1;
		}

		[enc dispatchThreads:MTLSizeMake((NSUInteger)T, (NSUInteger)T, 1)
		 threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
		[enc endEncoding];

		rc = hawkes_wait(cb);

		if (rc != 0) {
			return;
		}

		memcpy(out, [buf_o contents], ob);
	});

	return rc;
}

int metal_hawkes_kernel_matrix_tensor(
	const void *times,
	const void *alpha,
	const void *beta,
	void *out,
	int T) {
	if (!gHInited) {
		return -3;
	}

	if (!times || !alpha || !beta || !out || T <= 0) {
		return -1;
	}

	__block int rc = 0;

	hawkes_ensure_serial();
	dispatch_sync(gHSerial, ^{
		id<MTLBuffer> buf_t = (__bridge id)((void*)times);
		id<MTLBuffer> buf_alpha = (__bridge id)((void*)alpha);
		id<MTLBuffer> buf_beta = (__bridge id)((void*)beta);
		id<MTLBuffer> buf_o = (__bridge id)out;

		if (!buf_t || !buf_alpha || !buf_beta || !buf_o) {
			rc = -1;
			return;
		}

		id<MTLCommandBuffer> cb = [gHQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		if (!cb || !enc) {
			rc = -1;
			return;
		}

		[enc setComputePipelineState:gPSO_kmat];
		[enc setBuffer:buf_t offset:0 atIndex:0];
		[enc setBuffer:buf_o offset:0 atIndex:1];
		uint tu = (uint)T;
		[enc setBytes:&tu length:sizeof(tu) atIndex:2];
		[enc setBuffer:buf_alpha offset:0 atIndex:3];
		[enc setBuffer:buf_beta offset:0 atIndex:4];

		NSUInteger tx = T < 16 ? (NSUInteger)T : 16;
		NSUInteger ty = T < 16 ? (NSUInteger)T : 16;

		if (tx < 1) {
			tx = 1;
		}

		if (ty < 1) {
			ty = 1;
		}

		[enc dispatchThreads:MTLSizeMake((NSUInteger)T, (NSUInteger)T, 1)
		 threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
		[enc endEncoding];

		rc = hawkes_wait(cb);
	});

	return rc;
}

int metal_hawkes_log_likelihood(
	const float *intensities,
	float integral,
	float *out,
	int T) {
	if (!gHInited) {
		return -3;
	}

	if (!intensities || !out || T <= 0) {
		return -1;
	}

	__block int rc = 0;

	hawkes_ensure_serial();
	dispatch_sync(gHSerial, ^{
		size_t         bytes = (size_t)T * sizeof(float);
		id<MTLBuffer> buf_i = [gHDevice newBufferWithBytes:intensities
		                                              length:bytes
		                                             options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_p = [gHDevice newBufferWithLength:bytes
		                                           options:MTLResourceStorageModeShared];
		float          sum_zero = 0.f;
		id<MTLBuffer> buf_sum = [gHDevice newBufferWithBytes:&sum_zero
		                                              length:sizeof(float)
		                                             options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_out = [gHDevice newBufferWithLength:sizeof(float)
		                                            options:MTLResourceStorageModeShared];

		if (!buf_i || !buf_p || !buf_sum || !buf_out) {
			rc = -1;
			return;
		}

		id<MTLCommandBuffer>         cb   = [gHQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc0 = [cb computeCommandEncoder];

		[enc0 setComputePipelineState:gPSO_logt];
		[enc0 setBuffer:buf_i offset:0 atIndex:0];
		[enc0 setBuffer:buf_p offset:0 atIndex:1];
		uint tu = (uint)T;
		[enc0 setBytes:&tu length:sizeof(tu) atIndex:2];

		NSUInteger tw0 = gPSO_logt.threadExecutionWidth;

		[enc0 dispatchThreads:MTLSizeMake((NSUInteger)T, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(tw0, 1, 1)];
		[enc0 endEncoding];

		id<MTLComputeCommandEncoder> enc1 = [cb computeCommandEncoder];

		[enc1 setComputePipelineState:gPSO_logred];
		[enc1 setBuffer:buf_p offset:0 atIndex:0];
		[enc1 setBuffer:buf_sum offset:0 atIndex:1];
		[enc1 setBytes:&tu length:sizeof(tu) atIndex:2];

		NSUInteger tw1 = gPSO_logred.threadExecutionWidth;

		[enc1 dispatchThreads:MTLSizeMake((NSUInteger)T, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(tw1, 1, 1)];
		[enc1 endEncoding];

		id<MTLComputeCommandEncoder> enc2 = [cb computeCommandEncoder];

		[enc2 setComputePipelineState:gPSO_logfin];
		[enc2 setBuffer:buf_sum offset:0 atIndex:0];
		[enc2 setBytes:&integral length:sizeof(integral) atIndex:1];
		[enc2 setBuffer:buf_out offset:0 atIndex:2];

		[enc2 dispatchThreads:MTLSizeMake(1, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc2 endEncoding];

		rc = hawkes_wait(cb);

		if (rc != 0) {
			return;
		}

		memcpy(out, [buf_out contents], sizeof(float));
	});

	return rc;
}

int metal_hawkes_log_likelihood_tensor(
	const void *intensities,
	const void *integral,
	void *out,
	int T) {
	if (!gHInited) {
		return -3;
	}

	if (!intensities || !integral || !out || T <= 0) {
		return -1;
	}

	__block int rc = 0;

	hawkes_ensure_serial();
	dispatch_sync(gHSerial, ^{
		size_t bytes = (size_t)T * sizeof(float);
		id<MTLBuffer> buf_i = (__bridge id)((void*)intensities);
		id<MTLBuffer> buf_p = [gHDevice newBufferWithLength:bytes
		                                            options:MTLResourceStorageModeShared];
		float sum_zero = 0.f;
		id<MTLBuffer> buf_sum = [gHDevice newBufferWithBytes:&sum_zero
		                                               length:sizeof(float)
		                                              options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_integral = (__bridge id)((void*)integral);
		id<MTLBuffer> buf_out = (__bridge id)out;

		if (!buf_i || !buf_p || !buf_sum || !buf_integral || !buf_out) {
			rc = -1;
			return;
		}

		id<MTLCommandBuffer> cb = [gHQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc0 = [cb computeCommandEncoder];

		if (!cb || !enc0) {
			rc = -1;
			return;
		}

		[enc0 setComputePipelineState:gPSO_logt];
		[enc0 setBuffer:buf_i offset:0 atIndex:0];
		[enc0 setBuffer:buf_p offset:0 atIndex:1];
		uint tu = (uint)T;
		[enc0 setBytes:&tu length:sizeof(tu) atIndex:2];
		[enc0 dispatchThreads:MTLSizeMake((NSUInteger)T, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(gPSO_logt.threadExecutionWidth, 1, 1)];
		[enc0 endEncoding];

		id<MTLComputeCommandEncoder> enc1 = [cb computeCommandEncoder];
		[enc1 setComputePipelineState:gPSO_logred];
		[enc1 setBuffer:buf_p offset:0 atIndex:0];
		[enc1 setBuffer:buf_sum offset:0 atIndex:1];
		[enc1 setBytes:&tu length:sizeof(tu) atIndex:2];
		[enc1 dispatchThreads:MTLSizeMake((NSUInteger)T, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(gPSO_logred.threadExecutionWidth, 1, 1)];
		[enc1 endEncoding];

		id<MTLComputeCommandEncoder> enc2 = [cb computeCommandEncoder];
		[enc2 setComputePipelineState:gPSO_logfin];
		[enc2 setBuffer:buf_sum offset:0 atIndex:0];
		[enc2 setBuffer:buf_integral offset:0 atIndex:1];
		[enc2 setBuffer:buf_out offset:0 atIndex:2];
		[enc2 dispatchThreads:MTLSizeMake(1, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc2 endEncoding];

		rc = hawkes_wait(cb);
	});

	return rc;
}

int metal_hawkes_simulate(
	const float *mu, const float *alpha,
	const float *beta,
	float T_max, int K, int maxSteps,
	float *out) {
	if (!gHInited) {
		return -3;
	}

	if (!mu || !alpha || !beta || !out || K <= 0 || maxSteps <= 0 || T_max <= 0.f) {
		return -1;
	}

	if ((size_t)K * (size_t)maxSteps > (size_t)UINT_MAX) {
		return -1;
	}

	hawkes_ensure_serial();
	__block int rc = 0;

	dispatch_sync(gHSerial, ^{
		size_t        kbytes  = (size_t)K * sizeof(float);
		size_t        obytes  = (size_t)K * (size_t)maxSteps * sizeof(float);
		uint32_t      total_u = (uint32_t)((size_t)K * (size_t)maxSteps);
		id<MTLBuffer> buf_mu   = [gHDevice newBufferWithBytes:mu
		                                                length:kbytes
		                                               options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_al   = [gHDevice newBufferWithBytes:alpha
		                                                length:kbytes
		                                               options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_be   = [gHDevice newBufferWithBytes:beta
		                                                length:kbytes
		                                               options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_out  = [gHDevice newBufferWithLength:obytes
		                                                options:MTLResourceStorageModeShared];

		if (!buf_mu || !buf_al || !buf_be || !buf_out) {
			rc = -1;
			return;
		}

		id<MTLCommandBuffer>         cb   = [gHQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc0 = [cb computeCommandEncoder];

		[enc0 setComputePipelineState:gPSO_simclr];
		[enc0 setBuffer:buf_out offset:0 atIndex:0];
		[enc0 setBytes:&total_u length:sizeof(total_u) atIndex:1];

		NSUInteger twc = gPSO_simclr.threadExecutionWidth;

		[enc0 dispatchThreads:MTLSizeMake((NSUInteger)total_u, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(twc, 1, 1)];
		[enc0 endEncoding];

		id<MTLComputeCommandEncoder> enc1 = [cb computeCommandEncoder];

		[enc1 setComputePipelineState:gPSO_simdim];
		[enc1 setBuffer:buf_mu offset:0 atIndex:0];
		[enc1 setBuffer:buf_al offset:0 atIndex:1];
		[enc1 setBuffer:buf_be offset:0 atIndex:2];
		[enc1 setBuffer:buf_out offset:0 atIndex:3];
		[enc1 setBytes:&T_max length:sizeof(T_max) atIndex:4];
		uint ku = (uint)K;
		uint ms = (uint)maxSteps;
		[enc1 setBytes:&ku length:sizeof(ku) atIndex:5];
		[enc1 setBytes:&ms length:sizeof(ms) atIndex:6];

		NSUInteger twd = gPSO_simdim.threadExecutionWidth;

		[enc1 dispatchThreads:MTLSizeMake((NSUInteger)K, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(twd, 1, 1)];
		[enc1 endEncoding];

		rc = hawkes_wait(cb);

		if (rc != 0) {
			return;
		}

		memcpy(out, [buf_out contents], obytes);
	});

	return rc;
}

int metal_hawkes_simulate_tensor(
	const void *mu, const void *alpha,
	const void *beta,
	const void *T_max,
	int K, int maxSteps,
	void *out) {
	if (!gHInited) {
		return -3;
	}

	if (!mu || !alpha || !beta || !T_max || !out || K <= 0 || maxSteps <= 0) {
		return -1;
	}

	if ((size_t)K * (size_t)maxSteps > (size_t)UINT_MAX) {
		return -1;
	}

	hawkes_ensure_serial();
	__block int rc = 0;

	dispatch_sync(gHSerial, ^{
		uint32_t total_u = (uint32_t)((size_t)K * (size_t)maxSteps);
		id<MTLBuffer> buf_mu = (__bridge id)((void*)mu);
		id<MTLBuffer> buf_al = (__bridge id)((void*)alpha);
		id<MTLBuffer> buf_be = (__bridge id)((void*)beta);
		id<MTLBuffer> buf_tmax = (__bridge id)((void*)T_max);
		id<MTLBuffer> buf_out = (__bridge id)out;

		if (!buf_mu || !buf_al || !buf_be || !buf_tmax || !buf_out) {
			rc = -1;
			return;
		}

		id<MTLCommandBuffer> cb = [gHQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc0 = [cb computeCommandEncoder];

		if (!cb || !enc0) {
			rc = -1;
			return;
		}

		[enc0 setComputePipelineState:gPSO_simclr];
		[enc0 setBuffer:buf_out offset:0 atIndex:0];
		[enc0 setBytes:&total_u length:sizeof(total_u) atIndex:1];
		[enc0 dispatchThreads:MTLSizeMake((NSUInteger)total_u, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(gPSO_simclr.threadExecutionWidth, 1, 1)];
		[enc0 endEncoding];

		id<MTLComputeCommandEncoder> enc1 = [cb computeCommandEncoder];
		[enc1 setComputePipelineState:gPSO_simdim];
		[enc1 setBuffer:buf_mu offset:0 atIndex:0];
		[enc1 setBuffer:buf_al offset:0 atIndex:1];
		[enc1 setBuffer:buf_be offset:0 atIndex:2];
		[enc1 setBuffer:buf_out offset:0 atIndex:3];
		[enc1 setBuffer:buf_tmax offset:0 atIndex:4];
		uint ku = (uint)K;
		uint ms = (uint)maxSteps;
		[enc1 setBytes:&ku length:sizeof(ku) atIndex:5];
		[enc1 setBytes:&ms length:sizeof(ms) atIndex:6];
		[enc1 dispatchThreads:MTLSizeMake((NSUInteger)K, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(gPSO_simdim.threadExecutionWidth, 1, 1)];
		[enc1 endEncoding];

		rc = hawkes_wait(cb);
	});

	return rc;
}
