#include <math.h>
#include <stddef.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "active_inference.h"

#define METAL_AI_OK 0
#define METAL_AI_ERR_INVALID (-1)
#define METAL_AI_ERR_NOT_INIT (-2)
#define METAL_AI_ERR_NUMERIC (-3)

static id<MTLDevice>                    gAIDevice = nil;
static id<MTLCommandQueue>              gAIQueue  = nil;
static id<MTLComputePipelineState>     gPSO_fe   = nil;
static id<MTLComputePipelineState>     gPSO_fe_red = nil;
static id<MTLComputePipelineState>     gPSO_fe_fin = nil;
static id<MTLComputePipelineState>     gPSO_fe_serial = nil;
static id<MTLComputePipelineState>     gPSO_belief = nil;
static id<MTLComputePipelineState>     gPSO_pw   = nil;
static id<MTLComputePipelineState>     gPSO_efe  = nil;
static int                             gAIInited = 0;
static dispatch_queue_t                gAISerial = NULL;

static void ai_ensure_serial(void) {
	static dispatch_once_t onceToken;

	dispatch_once(&onceToken, ^{
		gAISerial = dispatch_queue_create("com.caramba.metal.active_inference", DISPATCH_QUEUE_SERIAL);
	});
}

static id<MTLComputePipelineState> ai_make_pso(
	id<MTLDevice> device, id<MTLLibrary> library, NSString *name
) {
	NSError *error   = nil;
	id<MTLFunction> fn = [library newFunctionWithName:name];

	if (!fn) {
		return nil;
	}

	return [device newComputePipelineStateWithFunction:fn error:&error];
}

static int ai_wait_command_buffer(id<MTLCommandBuffer> command_buffer) {
	dispatch_semaphore_t done = dispatch_semaphore_create(0);

	if (!done) {
		return METAL_AI_ERR_INVALID;
	}

	[command_buffer addCompletedHandler:^(id<MTLCommandBuffer> _) {
		dispatch_semaphore_signal(done);
	}];
	[command_buffer commit];
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);

	if (command_buffer.error != nil) {
		return METAL_AI_ERR_NUMERIC;
	}

	return METAL_AI_OK;
}

int metal_ai_init(const char *metallib_path) {
	if (metallib_path == NULL || metallib_path[0] == '\0') {
		return METAL_AI_ERR_INVALID;
	}

	ai_ensure_serial();
	__block int result = METAL_AI_OK;

	dispatch_sync(gAISerial, ^{
		if (gAIInited) {
			result = METAL_AI_OK;
			return;
		}

		gAIDevice = MTLCreateSystemDefaultDevice();

		if (!gAIDevice) {
			result = METAL_AI_ERR_INVALID;
			return;
		}

		gAIQueue = [gAIDevice newCommandQueue];

		if (!gAIQueue) {
			result = METAL_AI_ERR_INVALID;
			return;
		}

		NSString *path         = [NSString stringWithUTF8String:metallib_path];
		NSError *library_error = nil;
		id<MTLLibrary> library = [gAIDevice newLibraryWithURL:[NSURL fileURLWithPath:path]
		                                                  error:&library_error];

		if (!library) {
			result = METAL_AI_ERR_INVALID;
			return;
		}

		gPSO_fe     = ai_make_pso(gAIDevice, library, @"ai_free_energy_terms_kernel");
		gPSO_fe_red = ai_make_pso(gAIDevice, library, @"ai_reduce_fe_atomic_kernel");
		gPSO_fe_fin = ai_make_pso(gAIDevice, library, @"ai_finalize_free_energy_kernel");
		gPSO_fe_serial = ai_make_pso(gAIDevice, library, @"ai_free_energy_serial_kernel");
		gPSO_belief = ai_make_pso(gAIDevice, library, @"ai_belief_update_kernel");
		gPSO_pw     = ai_make_pso(gAIDevice, library, @"ai_precision_weight_kernel");
		gPSO_efe    = ai_make_pso(gAIDevice, library, @"ai_expected_free_energy_kernel");

		if (!gPSO_fe || !gPSO_fe_red || !gPSO_fe_fin || !gPSO_fe_serial || !gPSO_belief || !gPSO_pw || !gPSO_efe) {
			result = METAL_AI_ERR_INVALID;
			return;
		}

		gAIInited = 1;
	});

	return result;
}

int metal_ai_cleanup(void) {
	ai_ensure_serial();
	dispatch_sync(gAISerial, ^{
		gPSO_fe     = nil;
		gPSO_fe_red = nil;
		gPSO_fe_fin = nil;
		gPSO_fe_serial = nil;
		gPSO_belief = nil;
		gPSO_pw     = nil;
		gPSO_efe    = nil;
		gAIQueue    = nil;
		gAIDevice   = nil;
		gAIInited   = 0;
	});

	return METAL_AI_OK;
}

int metal_ai_free_energy(const float *mu, const float *log_sigma, float *out, int n) {
	if (!gAIInited) {
		return METAL_AI_ERR_NOT_INIT;
	}

	if (!out || n < 0) {
		return METAL_AI_ERR_INVALID;
	}

	if (n == 0) {
		out[0] = 0.f;
		return METAL_AI_OK;
	}

	if (!mu || !log_sigma) {
		return METAL_AI_ERR_INVALID;
	}

	__block int rc = METAL_AI_OK;

	ai_ensure_serial();
	dispatch_sync(gAISerial, ^{
		size_t         bytes = (size_t)n * sizeof(float);
		id<MTLBuffer> buf_mu = [gAIDevice newBufferWithBytes:mu
		                                              length:bytes
		                                             options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_ls = [gAIDevice newBufferWithBytes:log_sigma
		                                              length:bytes
		                                             options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_terms = [gAIDevice newBufferWithLength:bytes
		                                                  options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_sum = [gAIDevice newBufferWithLength:sizeof(float)
		                                                options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_outg = [gAIDevice newBufferWithLength:sizeof(float)
		                                                 options:MTLResourceStorageModeShared];

		if (!buf_mu || !buf_ls || !buf_terms || !buf_sum || !buf_outg) {
			rc = METAL_AI_ERR_INVALID;
			return;
		}

		id<MTLCommandBuffer>         command_buffer = [gAIQueue commandBuffer];
		id<MTLBlitCommandEncoder>    blit           = [command_buffer blitCommandEncoder];

		[blit fillBuffer:buf_sum range:NSMakeRange(0, sizeof(float)) value:0];
		[blit endEncoding];

		uint n_u = (uint)n;

		id<MTLComputeCommandEncoder> enc0 = [command_buffer computeCommandEncoder];

		[enc0 setComputePipelineState:gPSO_fe];
		[enc0 setBuffer:buf_mu    offset:0 atIndex:0];
		[enc0 setBuffer:buf_ls    offset:0 atIndex:1];
		[enc0 setBuffer:buf_terms offset:0 atIndex:2];
		[enc0 setBytes:&n_u length:sizeof(uint) atIndex:3];
		NSUInteger width0 = gPSO_fe.threadExecutionWidth;

		[enc0 dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(width0, 1, 1)];
		[enc0 endEncoding];

		id<MTLComputeCommandEncoder> enc1 = [command_buffer computeCommandEncoder];

		[enc1 setComputePipelineState:gPSO_fe_red];
		[enc1 setBuffer:buf_terms offset:0 atIndex:0];
		[enc1 setBuffer:buf_sum   offset:0 atIndex:1];
		[enc1 setBytes:&n_u length:sizeof(uint) atIndex:2];
		NSUInteger width1 = gPSO_fe_red.threadExecutionWidth;

		[enc1 dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(width1, 1, 1)];
		[enc1 endEncoding];

		id<MTLComputeCommandEncoder> enc2 = [command_buffer computeCommandEncoder];

		[enc2 setComputePipelineState:gPSO_fe_fin];
		[enc2 setBuffer:buf_sum  offset:0 atIndex:0];
		[enc2 setBuffer:buf_outg offset:0 atIndex:1];
		[enc2 dispatchThreads:MTLSizeMake(1, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc2 endEncoding];

		rc = ai_wait_command_buffer(command_buffer);

		if (rc != METAL_AI_OK) {
			return;
		}

		memcpy(out, [buf_outg contents], sizeof(float));
	});

	return rc;
}

int metal_ai_belief_update(
	const float *mu, const float *log_sigma,
	const float *pred_err, float lr,
	float *out, int n) {
	if (!gAIInited) {
		return METAL_AI_ERR_NOT_INIT;
	}

	if (!mu || !log_sigma || !pred_err || !out || n <= 0) {
		return METAL_AI_ERR_INVALID;
	}

	__block int rc = METAL_AI_OK;

	ai_ensure_serial();
	dispatch_sync(gAISerial, ^{
		size_t         bytes  = (size_t)n * sizeof(float);
		id<MTLBuffer> buf_mu  = [gAIDevice newBufferWithBytes:mu
		                                               length:bytes
		                                              options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_ls  = [gAIDevice newBufferWithBytes:log_sigma
		                                               length:bytes
		                                              options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_pe  = [gAIDevice newBufferWithBytes:pred_err
		                                               length:bytes
		                                              options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_out = [gAIDevice newBufferWithLength:(NSUInteger)(2 * bytes)
		                                                options:MTLResourceStorageModeShared];

		if (!buf_mu || !buf_ls || !buf_pe || !buf_out) {
			rc = METAL_AI_ERR_INVALID;
			return;
		}

		id<MTLCommandBuffer>         command_buffer = [gAIQueue commandBuffer];
		id<MTLComputeCommandEncoder> encoder        = [command_buffer computeCommandEncoder];

		[encoder setComputePipelineState:gPSO_belief];
		[encoder setBuffer:buf_mu  offset:0 atIndex:0];
		[encoder setBuffer:buf_ls  offset:0 atIndex:1];
		[encoder setBuffer:buf_pe  offset:0 atIndex:2];
		[encoder setBuffer:buf_out offset:0 atIndex:3];
		[encoder setBytes:&lr length:sizeof(lr) atIndex:4];
		uint n_u = (uint)n;
		[encoder setBytes:&n_u length:sizeof(uint) atIndex:5];

		NSUInteger width = gPSO_belief.threadExecutionWidth;

		[encoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
		[encoder endEncoding];

		rc = ai_wait_command_buffer(command_buffer);

		if (rc != METAL_AI_OK) {
			return;
		}

		memcpy(out, [buf_out contents], (size_t)(2 * n) * sizeof(float));
	});

	return rc;
}

int metal_ai_precision_weight(
	const float *err, const float *log_prec, float *out, int n) {
	if (!gAIInited) {
		return METAL_AI_ERR_NOT_INIT;
	}

	if (!err || !log_prec || !out || n <= 0) {
		return METAL_AI_ERR_INVALID;
	}

	for (int idx = 0; idx < n; idx++) {
		if (!isfinite((double)log_prec[idx]) || !isfinite((double)err[idx])) {
			return METAL_AI_ERR_NUMERIC;
		}
	}

	__block int rc = METAL_AI_OK;

	ai_ensure_serial();
	dispatch_sync(gAISerial, ^{
		size_t         bytes = (size_t)n * sizeof(float);
		id<MTLBuffer> buf_e  = [gAIDevice newBufferWithBytes:err
		                                              length:bytes
		                                             options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_lp = [gAIDevice newBufferWithBytes:log_prec
		                                              length:bytes
		                                             options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_o  = [gAIDevice newBufferWithLength:bytes
		                                             options:MTLResourceStorageModeShared];

		if (!buf_e || !buf_lp || !buf_o) {
			rc = METAL_AI_ERR_INVALID;
			return;
		}

		id<MTLCommandBuffer>         command_buffer = [gAIQueue commandBuffer];
		id<MTLComputeCommandEncoder> encoder        = [command_buffer computeCommandEncoder];

		[encoder setComputePipelineState:gPSO_pw];
		[encoder setBuffer:buf_e  offset:0 atIndex:0];
		[encoder setBuffer:buf_lp offset:0 atIndex:1];
		[encoder setBuffer:buf_o  offset:0 atIndex:2];
		uint n_u = (uint)n;
		[encoder setBytes:&n_u length:sizeof(uint) atIndex:3];

		NSUInteger width = gPSO_pw.threadExecutionWidth;

		[encoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
		[encoder endEncoding];

		rc = ai_wait_command_buffer(command_buffer);

		if (rc != METAL_AI_OK) {
			return;
		}

		memcpy(out, [buf_o contents], bytes);
	});

	return rc;
}

int metal_ai_expected_free_energy(
	const float *q_outcomes, float *out, int n, int K, float eps) {
	if (!gAIInited) {
		return METAL_AI_ERR_NOT_INIT;
	}

	if (!out || n < 0 || K < 0) {
		return METAL_AI_ERR_INVALID;
	}

	if (K == 0) {
		return METAL_AI_OK;
	}

	if (n == 0) {
		for (int k = 0; k < K; k++) {
			out[k] = 0.f;
		}

		return METAL_AI_OK;
	}

	if (!q_outcomes) {
		return METAL_AI_ERR_INVALID;
	}

	if (!(eps > 0.f) || !isfinite((double)eps)) {
		return METAL_AI_ERR_INVALID;
	}

	for (int row = 0; row < n; row++) {
		for (int k = 0; k < K; k++) {
			float q = q_outcomes[row * K + k];

			if (!isfinite((double)q)) {
				return METAL_AI_ERR_NUMERIC;
			}
		}
	}

	__block int rc = METAL_AI_OK;

	ai_ensure_serial();
	dispatch_sync(gAISerial, ^{
		size_t        qb    = (size_t)n * (size_t)K * sizeof(float);
		id<MTLBuffer> buf_q = [gAIDevice newBufferWithBytes:q_outcomes
		                                            length:qb
		                                           options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_o = [gAIDevice newBufferWithLength:(NSUInteger)K * sizeof(float)
		                                           options:MTLResourceStorageModeShared];

		if (!buf_q || !buf_o) {
			rc = METAL_AI_ERR_INVALID;
			return;
		}

		id<MTLCommandBuffer>         command_buffer = [gAIQueue commandBuffer];
		id<MTLComputeCommandEncoder> encoder        = [command_buffer computeCommandEncoder];

		[encoder setComputePipelineState:gPSO_efe];
		[encoder setBuffer:buf_q offset:0 atIndex:0];
		[encoder setBuffer:buf_o offset:0 atIndex:1];
		uint n_u = (uint)n;
		uint k_u = (uint)K;
		[encoder setBytes:&n_u length:sizeof(uint) atIndex:2];
		[encoder setBytes:&k_u length:sizeof(uint) atIndex:3];
		[encoder setBytes:&eps length:sizeof(eps) atIndex:4];

		NSUInteger width = gPSO_efe.threadExecutionWidth;

		[encoder dispatchThreads:MTLSizeMake((NSUInteger)K, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
		[encoder endEncoding];

		rc = ai_wait_command_buffer(command_buffer);

		if (rc != METAL_AI_OK) {
			return;
		}

		memcpy(out, [buf_o contents], (size_t)K * sizeof(float));
	});

	return rc;
}

int metal_ai_free_energy_tensor(const void *mu, const void *log_sigma, void *out, int n) {
	if (!gAIInited) {
		return METAL_AI_ERR_NOT_INIT;
	}

	if (!mu || !log_sigma || !out || n <= 0 || !gPSO_fe_serial) {
		return METAL_AI_ERR_INVALID;
	}

	__block int rc = METAL_AI_OK;

	ai_ensure_serial();
	dispatch_sync(gAISerial, ^{
		id<MTLBuffer> buf_mu = (__bridge id)((void*)mu);
		id<MTLBuffer> buf_ls = (__bridge id)((void*)log_sigma);
		id<MTLBuffer> buf_out = (__bridge id)out;

		if (!buf_mu || !buf_ls || !buf_out) {
			rc = METAL_AI_ERR_INVALID;
			return;
		}

		id<MTLCommandBuffer> command_buffer = [gAIQueue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

		if (!command_buffer || !encoder) {
			rc = METAL_AI_ERR_INVALID;
			return;
		}

		uint n_u = (uint)n;
		[encoder setComputePipelineState:gPSO_fe_serial];
		[encoder setBuffer:buf_mu offset:0 atIndex:0];
		[encoder setBuffer:buf_ls offset:0 atIndex:1];
		[encoder setBuffer:buf_out offset:0 atIndex:2];
		[encoder setBytes:&n_u length:sizeof(uint) atIndex:3];
		[encoder dispatchThreads:MTLSizeMake(1, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[encoder endEncoding];

		rc = ai_wait_command_buffer(command_buffer);
	});

	return rc;
}

int metal_ai_belief_update_tensor(
	const void *mu, const void *log_sigma,
	const void *pred_err, float lr,
	void *out, int n) {
	if (!gAIInited) {
		return METAL_AI_ERR_NOT_INIT;
	}

	if (!mu || !log_sigma || !pred_err || !out || n <= 0) {
		return METAL_AI_ERR_INVALID;
	}

	__block int rc = METAL_AI_OK;

	ai_ensure_serial();
	dispatch_sync(gAISerial, ^{
		id<MTLBuffer> buf_mu = (__bridge id)((void*)mu);
		id<MTLBuffer> buf_ls = (__bridge id)((void*)log_sigma);
		id<MTLBuffer> buf_pe = (__bridge id)((void*)pred_err);
		id<MTLBuffer> buf_out = (__bridge id)out;

		if (!buf_mu || !buf_ls || !buf_pe || !buf_out) {
			rc = METAL_AI_ERR_INVALID;
			return;
		}

		id<MTLCommandBuffer> command_buffer = [gAIQueue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

		[encoder setComputePipelineState:gPSO_belief];
		[encoder setBuffer:buf_mu offset:0 atIndex:0];
		[encoder setBuffer:buf_ls offset:0 atIndex:1];
		[encoder setBuffer:buf_pe offset:0 atIndex:2];
		[encoder setBuffer:buf_out offset:0 atIndex:3];
		[encoder setBytes:&lr length:sizeof(lr) atIndex:4];
		uint n_u = (uint)n;
		[encoder setBytes:&n_u length:sizeof(uint) atIndex:5];
		[encoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(gPSO_belief.threadExecutionWidth, 1, 1)];
		[encoder endEncoding];

		rc = ai_wait_command_buffer(command_buffer);
	});

	return rc;
}

int metal_ai_precision_weight_tensor(
	const void *err, const void *log_prec, void *out, int n) {
	if (!gAIInited) {
		return METAL_AI_ERR_NOT_INIT;
	}

	if (!err || !log_prec || !out || n <= 0) {
		return METAL_AI_ERR_INVALID;
	}

	__block int rc = METAL_AI_OK;

	ai_ensure_serial();
	dispatch_sync(gAISerial, ^{
		id<MTLBuffer> buf_e = (__bridge id)((void*)err);
		id<MTLBuffer> buf_lp = (__bridge id)((void*)log_prec);
		id<MTLBuffer> buf_o = (__bridge id)out;

		if (!buf_e || !buf_lp || !buf_o) {
			rc = METAL_AI_ERR_INVALID;
			return;
		}

		id<MTLCommandBuffer> command_buffer = [gAIQueue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

		[encoder setComputePipelineState:gPSO_pw];
		[encoder setBuffer:buf_e offset:0 atIndex:0];
		[encoder setBuffer:buf_lp offset:0 atIndex:1];
		[encoder setBuffer:buf_o offset:0 atIndex:2];
		uint n_u = (uint)n;
		[encoder setBytes:&n_u length:sizeof(uint) atIndex:3];
		[encoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(gPSO_pw.threadExecutionWidth, 1, 1)];
		[encoder endEncoding];

		rc = ai_wait_command_buffer(command_buffer);
	});

	return rc;
}

int metal_ai_expected_free_energy_tensor(
	const void *q_outcomes, void *out, int n, int K, float eps) {
	if (!gAIInited) {
		return METAL_AI_ERR_NOT_INIT;
	}

	if (!q_outcomes || !out || n <= 0 || K <= 0 || !(eps > 0.f) || !isfinite((double)eps)) {
		return METAL_AI_ERR_INVALID;
	}

	__block int rc = METAL_AI_OK;

	ai_ensure_serial();
	dispatch_sync(gAISerial, ^{
		id<MTLBuffer> buf_q = (__bridge id)((void*)q_outcomes);
		id<MTLBuffer> buf_o = (__bridge id)out;

		if (!buf_q || !buf_o) {
			rc = METAL_AI_ERR_INVALID;
			return;
		}

		id<MTLCommandBuffer> command_buffer = [gAIQueue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

		[encoder setComputePipelineState:gPSO_efe];
		[encoder setBuffer:buf_q offset:0 atIndex:0];
		[encoder setBuffer:buf_o offset:0 atIndex:1];
		uint n_u = (uint)n;
		uint k_u = (uint)K;
		[encoder setBytes:&n_u length:sizeof(uint) atIndex:2];
		[encoder setBytes:&k_u length:sizeof(uint) atIndex:3];
		[encoder setBytes:&eps length:sizeof(eps) atIndex:4];
		[encoder dispatchThreads:MTLSizeMake((NSUInteger)K, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(gPSO_efe.threadExecutionWidth, 1, 1)];
		[encoder endEncoding];

		rc = ai_wait_command_buffer(command_buffer);
	});

	return rc;
}
