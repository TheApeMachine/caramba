#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stddef.h>
#include "predictive_coding.h"

static id<MTLDevice>                gPCDevice = nil;
static id<MTLCommandQueue>          gPCQueue  = nil;
static id<MTLComputePipelineState> gPSO_pred = nil;
static id<MTLComputePipelineState> gPSO_perr = nil;
static id<MTLComputePipelineState> gPSO_urep = nil;
static id<MTLComputePipelineState> gPSO_uw   = nil;
static int                          gPCInited = 0;
static dispatch_queue_t             gPCSerial = NULL;

static void pc_ensure_serial(void) {
	static dispatch_once_t onceToken;

	dispatch_once(&onceToken, ^{
		gPCSerial = dispatch_queue_create("com.caramba.metal.predictive_coding", DISPATCH_QUEUE_SERIAL);
	});
}

static id<MTLComputePipelineState> pc_make_pso(
	id<MTLDevice> device, id<MTLLibrary> library, NSString *name
) {
	NSError *err = nil;
	id<MTLFunction> fn = [library newFunctionWithName:name];

	if (!fn) {
		return nil;
	}

	return [device newComputePipelineStateWithFunction:fn error:&err];
}

static int pc_wait(id<MTLCommandBuffer> cb) {
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

int metal_pc_init(const char *metallib_path) {
	if (metallib_path == NULL || metallib_path[0] == '\0') {
		return -1;
	}

	pc_ensure_serial();
	__block int result = 0;

	dispatch_sync(gPCSerial, ^{
		if (gPCInited) {
			return;
		}

		gPCDevice = MTLCreateSystemDefaultDevice();

		if (!gPCDevice) {
			result = -1;
			return;
		}

		gPCQueue = [gPCDevice newCommandQueue];

		if (!gPCQueue) {
			result = -1;
			return;
		}

		NSString *path = [NSString stringWithUTF8String:metallib_path];
		NSError *err   = nil;
		id<MTLLibrary> lib = [gPCDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];

		if (!lib) {
			result = -1;
			return;
		}

		gPSO_pred = pc_make_pso(gPCDevice, lib, @"pc_prediction_kernel");
		gPSO_perr = pc_make_pso(gPCDevice, lib, @"pc_prediction_error_kernel");
		gPSO_urep = pc_make_pso(gPCDevice, lib, @"pc_update_representation_kernel");
		gPSO_uw   = pc_make_pso(gPCDevice, lib, @"pc_update_weights_kernel");

		if (!gPSO_pred || !gPSO_perr || !gPSO_urep || !gPSO_uw) {
			result = -1;
			return;
		}

		gPCInited = 1;
	});

	return result;
}

int metal_pc_shutdown(void) {
	pc_ensure_serial();
	dispatch_sync(gPCSerial, ^{
		gPSO_pred = nil;
		gPSO_perr = nil;
		gPSO_urep = nil;
		gPSO_uw   = nil;
		gPCQueue  = nil;
		gPCDevice = nil;
		gPCInited = 0;
	});

	return 0;
}

int metal_pc_prediction(const float *W, const float *r, float *dst, int D_out, int D_in) {
	if (!gPCInited) {
		return -3;
	}

	if (!W || !r || !dst || D_out <= 0 || D_in <= 0) {
		return -1;
	}

	__block int rc = 0;

	pc_ensure_serial();
	dispatch_sync(gPCSerial, ^{
		size_t wlen = (size_t)D_out * (size_t)D_in * sizeof(float);
		size_t rlen = (size_t)D_in * sizeof(float);
		size_t dlen = (size_t)D_out * sizeof(float);
		id<MTLBuffer> buf_w = [gPCDevice newBufferWithBytes:W
		                                             length:wlen
		                                            options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_r = [gPCDevice newBufferWithBytes:r
		                                             length:rlen
		                                            options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_d = [gPCDevice newBufferWithLength:dlen
		                                           options:MTLResourceStorageModeShared];

		if (!buf_w || !buf_r || !buf_d) {
			rc = -1;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gPCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_pred];
		[enc setBuffer:buf_w offset:0 atIndex:0];
		[enc setBuffer:buf_r offset:0 atIndex:1];
		[enc setBuffer:buf_d offset:0 atIndex:2];
		uint o = (uint)D_out;
		uint i = (uint)D_in;
		[enc setBytes:&o length:sizeof(o) atIndex:3];
		[enc setBytes:&i length:sizeof(i) atIndex:4];

		NSUInteger tw = gPSO_pred.threadExecutionWidth;

		[enc dispatchThreads:MTLSizeMake((NSUInteger)D_out, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
		[enc endEncoding];

		rc = pc_wait(cb);

		if (rc != 0) {
			return;
		}

		memcpy(dst, [buf_d contents], dlen);
	});

	return rc;
}

int metal_pc_prediction_error(
	const float *x, const float *mu_hat,
	const float *prec, float *dst, int n) {
	if (!gPCInited) {
		return -3;
	}

	if (!x || !mu_hat || !dst || n <= 0) {
		return -1;
	}

	uint use_prec = prec ? 1u : 0u;

	__block int rc = 0;

	pc_ensure_serial();
	dispatch_sync(gPCSerial, ^{
		size_t         bytes = (size_t)n * sizeof(float);
		id<MTLBuffer> buf_x  = [gPCDevice newBufferWithBytes:x
		                                              length:bytes
		                                             options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_m  = [gPCDevice newBufferWithBytes:mu_hat
		                                              length:bytes
		                                             options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_p  = nil;

		if (prec) {
			buf_p = [gPCDevice newBufferWithBytes:prec
			                               length:bytes
			                              options:MTLResourceStorageModeShared];
		} else {
			buf_p = [gPCDevice newBufferWithLength:sizeof(float)
			                                options:MTLResourceStorageModeShared];
		}

		id<MTLBuffer> buf_d = [gPCDevice newBufferWithLength:bytes
		                                            options:MTLResourceStorageModeShared];

		if (!buf_x || !buf_m || !buf_p || !buf_d) {
			rc = -1;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gPCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_perr];
		[enc setBuffer:buf_x offset:0 atIndex:0];
		[enc setBuffer:buf_m offset:0 atIndex:1];
		[enc setBuffer:buf_p offset:0 atIndex:2];
		[enc setBuffer:buf_d offset:0 atIndex:3];
		uint nu = (uint)n;
		[enc setBytes:&nu length:sizeof(nu) atIndex:4];
		[enc setBytes:&use_prec length:sizeof(use_prec) atIndex:5];

		NSUInteger tw = gPSO_perr.threadExecutionWidth;

		[enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
		[enc endEncoding];

		rc = pc_wait(cb);

		if (rc != 0) {
			return;
		}

		memcpy(dst, [buf_d contents], bytes);
	});

	return rc;
}

int metal_pc_update_representation(
	const float *r, const float *W,
	const float *eps_lower, const float *eps_self,
	float lr, float *dst, int D_out, int D_in) {
	if (!gPCInited) {
		return -3;
	}

	if (!r || !W || !eps_lower || !eps_self || !dst || D_out <= 0 || D_in <= 0) {
		return -1;
	}

	__block int rc = 0;

	pc_ensure_serial();
	dispatch_sync(gPCSerial, ^{
		size_t rlen = (size_t)D_in * sizeof(float);
		size_t wlen = (size_t)D_out * (size_t)D_in * sizeof(float);
		size_t elen = (size_t)D_out * sizeof(float);
		id<MTLBuffer> buf_r  = [gPCDevice newBufferWithBytes:r
		                                              length:rlen
		                                             options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_w  = [gPCDevice newBufferWithBytes:W
		                                              length:wlen
		                                             options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_el = [gPCDevice newBufferWithBytes:eps_lower
		                                              length:elen
		                                             options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_es = [gPCDevice newBufferWithBytes:eps_self
		                                              length:rlen
		                                             options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_d  = [gPCDevice newBufferWithLength:rlen
		                                            options:MTLResourceStorageModeShared];

		if (!buf_r || !buf_w || !buf_el || !buf_es || !buf_d) {
			rc = -1;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gPCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_urep];
		[enc setBuffer:buf_r  offset:0 atIndex:0];
		[enc setBuffer:buf_w  offset:0 atIndex:1];
		[enc setBuffer:buf_el offset:0 atIndex:2];
		[enc setBuffer:buf_es offset:0 atIndex:3];
		[enc setBuffer:buf_d  offset:0 atIndex:4];
		[enc setBytes:&lr length:sizeof(lr) atIndex:5];
		uint o = (uint)D_out;
		uint i = (uint)D_in;
		[enc setBytes:&o length:sizeof(o) atIndex:6];
		[enc setBytes:&i length:sizeof(i) atIndex:7];

		NSUInteger tw = gPSO_urep.threadExecutionWidth;

		[enc dispatchThreads:MTLSizeMake((NSUInteger)D_in, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
		[enc endEncoding];

		rc = pc_wait(cb);

		if (rc != 0) {
			return;
		}

		memcpy(dst, [buf_d contents], rlen);
	});

	return rc;
}

int metal_pc_update_weights(
	const float *W, const float *eps, const float *r,
	float lr, float *dst, int D_out, int D_in) {
	if (!gPCInited) {
		return -3;
	}

	if (!W || !eps || !r || !dst || D_out <= 0 || D_in <= 0) {
		return -1;
	}

	__block int rc = 0;

	pc_ensure_serial();
	dispatch_sync(gPCSerial, ^{
		size_t wlen = (size_t)D_out * (size_t)D_in * sizeof(float);
		size_t elen = (size_t)D_out * sizeof(float);
		size_t rlen = (size_t)D_in * sizeof(float);
		id<MTLBuffer> buf_w = [gPCDevice newBufferWithBytes:W
		                                             length:wlen
		                                            options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_e = [gPCDevice newBufferWithBytes:eps
		                                             length:elen
		                                            options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_r = [gPCDevice newBufferWithBytes:r
		                                             length:rlen
		                                            options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_d = [gPCDevice newBufferWithLength:wlen
		                                           options:MTLResourceStorageModeShared];

		if (!buf_w || !buf_e || !buf_r || !buf_d) {
			rc = -1;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gPCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_uw];
		[enc setBuffer:buf_w offset:0 atIndex:0];
		[enc setBuffer:buf_e offset:0 atIndex:1];
		[enc setBuffer:buf_r offset:0 atIndex:2];
		[enc setBuffer:buf_d offset:0 atIndex:3];
		[enc setBytes:&lr length:sizeof(lr) atIndex:4];
		uint o = (uint)D_out;
		uint i = (uint)D_in;
		[enc setBytes:&o length:sizeof(o) atIndex:5];
		[enc setBytes:&i length:sizeof(i) atIndex:6];

		NSUInteger tx = 16;
		NSUInteger ty = 16;

		if (D_out < 16) {
			tx = (NSUInteger)D_out;
		}

		if (D_in < 16) {
			ty = (NSUInteger)D_in;
		}

		if (tx < 1) {
			tx = 1;
		}

		if (ty < 1) {
			ty = 1;
		}

		[enc dispatchThreads:MTLSizeMake((NSUInteger)D_out, (NSUInteger)D_in, 1)
		 threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
		[enc endEncoding];

		rc = pc_wait(cb);

		if (rc != 0) {
			return;
		}

		memcpy(dst, [buf_d contents], wlen);
	});

	return rc;
}
