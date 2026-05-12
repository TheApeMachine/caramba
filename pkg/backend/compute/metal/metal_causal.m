#include <math.h>
#include <stddef.h>
#include <string.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "causal.h"

static id<MTLDevice>                gCDevice = nil;
static id<MTLCommandQueue>          gCQueue  = nil;
static id<MTLComputePipelineState> gPSO_axpy = nil;
static id<MTLComputePipelineState> gPSO_sub  = nil;
static id<MTLComputePipelineState> gPSO_matvec = nil;
static id<MTLComputePipelineState> gPSO_dotatom = nil;
static id<MTLComputePipelineState> gPSO_docalc = nil;
static id<MTLComputePipelineState> gPSO_backdoor = nil;
static id<MTLComputePipelineState> gPSO_iv = nil;
static id<MTLComputePipelineState> gPSO_cate = nil;
static id<MTLComputePipelineState> gPSO_dag = nil;
static int                          gCInited = 0;
static dispatch_queue_t             gCSerial = NULL;

static void c_ensure_serial(void) {
	static dispatch_once_t onceToken;

	dispatch_once(&onceToken, ^{
		gCSerial = dispatch_queue_create("com.caramba.metal.causal", DISPATCH_QUEUE_SERIAL);
	});
}

static id<MTLComputePipelineState> c_make_pso(
	id<MTLDevice> device, id<MTLLibrary> library, NSString *name
) {
	NSError *err = nil;
	id<MTLFunction> fn = [library newFunctionWithName:name];

	if (!fn) {
		return nil;
	}

	return [device newComputePipelineStateWithFunction:fn error:&err];
}

static int c_wait(id<MTLCommandBuffer> cb) {
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

int metal_causal_init(const char *metallib_path) {
	if (metallib_path == NULL || metallib_path[0] == '\0') {
		return -1;
	}

	c_ensure_serial();
	__block int result = 0;

	dispatch_sync(gCSerial, ^{
		if (gCInited) {
			return;
		}

		gCDevice = MTLCreateSystemDefaultDevice();

		if (!gCDevice) {
			result = -1;
			return;
		}

		gCQueue = [gCDevice newCommandQueue];

		if (!gCQueue) {
			result = -1;
			return;
		}

		NSString *path = [NSString stringWithUTF8String:metallib_path];
		NSError *err   = nil;
		id<MTLLibrary> lib = [gCDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];

		if (!lib) {
			result = -1;
			return;
		}

		gPSO_axpy     = c_make_pso(gCDevice, lib, @"causal_axpy_kernel");
		gPSO_sub      = c_make_pso(gCDevice, lib, @"causal_sub_kernel");
		gPSO_matvec   = c_make_pso(gCDevice, lib, @"causal_matvec_kernel");
		gPSO_dotatom  = c_make_pso(gCDevice, lib, @"causal_dot_atomic_kernel");
		gPSO_docalc   = c_make_pso(gCDevice, lib, @"causal_do_calculus_kernel");
		gPSO_backdoor = c_make_pso(gCDevice, lib, @"causal_backdoor_kernel");
		gPSO_iv       = c_make_pso(gCDevice, lib, @"causal_iv_kernel");
		gPSO_cate     = c_make_pso(gCDevice, lib, @"causal_cate_kernel");
		gPSO_dag      = c_make_pso(gCDevice, lib, @"causal_dag_markov_kernel");

		if (
			!gPSO_axpy || !gPSO_sub || !gPSO_matvec || !gPSO_dotatom
			|| !gPSO_docalc || !gPSO_backdoor || !gPSO_iv || !gPSO_cate || !gPSO_dag
		) {
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
		gPSO_axpy     = nil;
		gPSO_sub      = nil;
		gPSO_matvec   = nil;
		gPSO_dotatom  = nil;
		gPSO_docalc   = nil;
		gPSO_backdoor = nil;
		gPSO_iv       = nil;
		gPSO_cate     = nil;
		gPSO_dag      = nil;
		gCQueue       = nil;
		gCDevice      = nil;
		gCInited      = 0;
	});

	return 0;
}

static int c_check(void) {
	return gCInited ? 0 : -3;
}

int metal_causal_axpy(float *dst, const float *src, float scale, int n) {
	if (c_check()) {
		return -3;
	}

	if (!dst || !src || n <= 0) {
		return -1;
	}

	__block int rc = 0;

	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		size_t        nb = (size_t)n * sizeof(float);
		id<MTLBuffer> buf_d = [gCDevice newBufferWithBytes:dst length:nb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_s = [gCDevice newBufferWithBytes:src length:nb options:MTLResourceStorageModeShared];

		if (!buf_d || !buf_s) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_axpy];
		[enc setBuffer:buf_d offset:0 atIndex:0];
		[enc setBuffer:buf_s offset:0 atIndex:1];
		[enc setBytes:&scale length:sizeof(scale) atIndex:2];
		int nu = n;
		[enc setBytes:&nu length:sizeof(nu) atIndex:3];

		NSUInteger tw = gPSO_axpy.threadExecutionWidth;

		[enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
		[enc endEncoding];

		rc = c_wait(cb);

		if (rc != 0) {
			return;
		}

		memcpy(dst, [buf_d contents], nb);
	});

	return rc;
}

int metal_causal_dot(const float *a, const float *b, float *out, int n) {
	if (c_check()) {
		return -3;
	}

	if (!a || !b || !out || n <= 0) {
		return -1;
	}

	__block int rc = 0;

	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		size_t        nb = (size_t)n * sizeof(float);
		id<MTLBuffer> buf_a = [gCDevice newBufferWithBytes:a length:nb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_b = [gCDevice newBufferWithBytes:b length:nb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_s = [gCDevice newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];

		if (!buf_a || !buf_b || !buf_s) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer>        cb   = [gCQueue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];

		[blit fillBuffer:buf_s range:NSMakeRange(0, sizeof(float)) value:0];
		[blit endEncoding];

		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_dotatom];
		[enc setBuffer:buf_a offset:0 atIndex:0];
		[enc setBuffer:buf_b offset:0 atIndex:1];
		[enc setBuffer:buf_s offset:0 atIndex:2];
		int nu = n;
		[enc setBytes:&nu length:sizeof(nu) atIndex:3];

		NSUInteger tw = gPSO_dotatom.threadExecutionWidth;

		[enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
		[enc endEncoding];

		rc = c_wait(cb);

		if (rc != 0) {
			return;
		}

		memcpy(out, [buf_s contents], sizeof(float));
	});

	return rc;
}

int metal_causal_sub(float *dst, const float *a, const float *b, int n) {
	if (c_check()) {
		return -3;
	}

	if (!dst || !a || !b || n <= 0) {
		return -1;
	}

	__block int rc = 0;

	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		size_t        nb = (size_t)n * sizeof(float);
		id<MTLBuffer> buf_d = [gCDevice newBufferWithLength:nb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_a = [gCDevice newBufferWithBytes:a length:nb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_b = [gCDevice newBufferWithBytes:b length:nb options:MTLResourceStorageModeShared];

		if (!buf_d || !buf_a || !buf_b) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_sub];
		[enc setBuffer:buf_d offset:0 atIndex:0];
		[enc setBuffer:buf_a offset:0 atIndex:1];
		[enc setBuffer:buf_b offset:0 atIndex:2];
		int nu = n;
		[enc setBytes:&nu length:sizeof(nu) atIndex:3];

		NSUInteger tw = gPSO_sub.threadExecutionWidth;

		[enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
		[enc endEncoding];

		rc = c_wait(cb);

		if (rc != 0) {
			return;
		}

		memcpy(dst, [buf_d contents], nb);
	});

	return rc;
}

int metal_causal_matvec(float *dst, const float *W, const float *x, int rows, int cols) {
	if (c_check()) {
		return -3;
	}

	if (!dst || !W || !x || rows <= 0 || cols <= 0) {
		return -1;
	}

	__block int rc = 0;

	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		size_t        wb = (size_t)rows * (size_t)cols * sizeof(float);
		size_t        xb = (size_t)cols * sizeof(float);
		size_t        db = (size_t)rows * sizeof(float);
		id<MTLBuffer> buf_d = [gCDevice newBufferWithLength:db options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_w = [gCDevice newBufferWithBytes:W length:wb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_x = [gCDevice newBufferWithBytes:x length:xb options:MTLResourceStorageModeShared];

		if (!buf_d || !buf_w || !buf_x) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_matvec];
		[enc setBuffer:buf_d offset:0 atIndex:0];
		[enc setBuffer:buf_w offset:0 atIndex:1];
		[enc setBuffer:buf_x offset:0 atIndex:2];
		int r0 = rows;
		int c0 = cols;
		[enc setBytes:&r0 length:sizeof(r0) atIndex:3];
		[enc setBytes:&c0 length:sizeof(c0) atIndex:4];

		NSUInteger tw = gPSO_matvec.threadExecutionWidth;

		[enc dispatchThreads:MTLSizeMake((NSUInteger)rows, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
		[enc endEncoding];

		rc = c_wait(cb);

		if (rc != 0) {
			return;
		}

		memcpy(dst, [buf_d contents], db);
	});

	return rc;
}

int metal_causal_do_calculus(
	const float *cov_f, const float *mask_f, const float *values_f,
	float *out_f, int N) {
	if (c_check()) {
		return -3;
	}

	if (!cov_f || !mask_f || !values_f || !out_f || N <= 0) {
		return -1;
	}

	__block int rc = 0;

	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		size_t wf =
			12 * (size_t)N * (size_t)N + 8 * (size_t)N;
		size_t wi = 2 * (size_t)N * sizeof(int);

		id<MTLBuffer> buf_cov = [gCDevice newBufferWithBytes:cov_f
		                                   length:(size_t)N * (size_t)N * sizeof(float)
		                                  options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_mask = [gCDevice newBufferWithBytes:mask_f length:(size_t)N * sizeof(float)
		                                                options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_val = [gCDevice newBufferWithBytes:values_f length:(size_t)N * sizeof(float)
		                                               options:MTLResourceStorageModeShared];
		size_t outLen = (size_t)N + (size_t)N * (size_t)N;
		id<MTLBuffer> buf_out = [gCDevice newBufferWithLength:outLen * sizeof(float)
		                                                options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_wf  = [gCDevice newBufferWithLength:wf * sizeof(float)
		                                               options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_wi  = [gCDevice newBufferWithLength:wi options:MTLResourceStorageModeShared];
		int           er0     = 0;
		id<MTLBuffer> buf_err = [gCDevice newBufferWithBytes:&er0 length:sizeof(int)
		                                                   options:MTLResourceStorageModeShared];

		if (!buf_cov || !buf_mask || !buf_val || !buf_out || !buf_wf || !buf_wi || !buf_err) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_docalc];
		[enc setBuffer:buf_cov offset:0 atIndex:0];
		[enc setBuffer:buf_mask offset:0 atIndex:1];
		[enc setBuffer:buf_val offset:0 atIndex:2];
		[enc setBuffer:buf_out offset:0 atIndex:3];
		[enc setBuffer:buf_wf offset:0 atIndex:4];
		[enc setBuffer:buf_wi offset:0 atIndex:5];
		int n0 = N;
		[enc setBytes:&n0 length:sizeof(n0) atIndex:6];
		[enc setBuffer:buf_err offset:0 atIndex:7];
		[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc endEncoding];

		rc = c_wait(cb);

		if (rc != 0) {
			return;
		}

		int ec = *((int *)[buf_err contents]);

		if (ec != 0) {
			rc = ec;
			return;
		}

		memcpy(out_f, [buf_out contents], outLen * sizeof(float));
	});

	return rc;
}

int metal_causal_backdoor(
	const float *Y, const float *X, const float *Z,
	float *effect,
	int T, int ny, int nx, int nz) {
	if (c_check()) {
		return -3;
	}

	if (!Y || !X || !Z || !effect || T <= 0 || ny <= 0 || nx <= 0 || nz < 0) {
		return -1;
	}

	int p = 1 + nx + nz;

	__block int rc = 0;

	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		size_t wf = (size_t)T * (size_t)p + 3 * (size_t)p * (size_t)p + 3 * (size_t)p;
		id<MTLBuffer> buf_y = [gCDevice newBufferWithBytes:Y
		                                            length:(size_t)T * (size_t)ny * sizeof(float)
		                                           options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_x = [gCDevice newBufferWithBytes:X
		                                            length:(size_t)T * (size_t)nx * sizeof(float)
		                                           options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_z = [gCDevice newBufferWithBytes:Z
		                                            length:(size_t)T * (size_t)nz * sizeof(float)
		                                           options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_e = [gCDevice newBufferWithLength:(size_t)ny * sizeof(float)
		                                           options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_w = [gCDevice newBufferWithLength:wf * sizeof(float)
		                                            options:MTLResourceStorageModeShared];
		int           er0 = 0;
		id<MTLBuffer> buf_err = [gCDevice newBufferWithBytes:&er0 length:sizeof(int)
		                                               options:MTLResourceStorageModeShared];

		if (!buf_y || !buf_x || !buf_z || !buf_e || !buf_w || !buf_err) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_backdoor];
		[enc setBuffer:buf_y offset:0 atIndex:0];
		[enc setBuffer:buf_x offset:0 atIndex:1];
		[enc setBuffer:buf_z offset:0 atIndex:2];
		[enc setBuffer:buf_e offset:0 atIndex:3];
		[enc setBuffer:buf_w offset:0 atIndex:4];
		[enc setBuffer:buf_err offset:0 atIndex:5];
		int t0 = T;
		int ny0 = ny;
		int nx0 = nx;
		int nz0 = nz;
		[enc setBytes:&t0 length:sizeof(t0) atIndex:6];
		[enc setBytes:&ny0 length:sizeof(ny0) atIndex:7];
		[enc setBytes:&nx0 length:sizeof(nx0) atIndex:8];
		[enc setBytes:&nz0 length:sizeof(nz0) atIndex:9];
		[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc endEncoding];

		rc = c_wait(cb);

		if (rc != 0) {
			return;
		}

		int ec = *((int *)[buf_err contents]);

		if (ec != 0) {
			rc = ec;
			return;
		}

		memcpy(effect, [buf_e contents], (size_t)ny * sizeof(float));
	});

	return rc;
}

int metal_causal_iv(
	const float *Z, const float *X, const float *Y,
	float *beta_iv,
	int T, int nz, int nx, int ny) {
	if (c_check()) {
		return -3;
	}

	if (!Z || !X || !Y || !beta_iv || T <= 0 || nz <= 0 || nx <= 0 || ny <= 0) {
		return -1;
	}

	__block int rc = 0;

	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		int           nmax = (nz > nx) ? nz : nx;
		unsigned long wf =
			2ul * (unsigned long)nz * (unsigned long)nz
			+ 2ul * (unsigned long)nz * (unsigned long)nx
			+ (unsigned long)T * (unsigned long)nx
			+ 2ul * (unsigned long)nx * (unsigned long)nx
			+ 2ul * (unsigned long)nx * (unsigned long)ny
			+ (unsigned long)nmax * (unsigned long)nmax
			+ 2ul * (unsigned long)nmax;

		id<MTLBuffer> buf_z = [gCDevice newBufferWithBytes:Z
		                                            length:(size_t)T * (size_t)nz * sizeof(float)
		                                           options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_x = [gCDevice newBufferWithBytes:X
		                                            length:(size_t)T * (size_t)nx * sizeof(float)
		                                           options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_y = [gCDevice newBufferWithBytes:Y
		                                            length:(size_t)T * (size_t)ny * sizeof(float)
		                                           options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_b = [gCDevice newBufferWithLength:(size_t)nx * (size_t)ny * sizeof(float)
		                                            options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_w = [gCDevice newBufferWithLength:(size_t)wf * sizeof(float)
		                                            options:MTLResourceStorageModeShared];
		int           er0 = 0;
		id<MTLBuffer> buf_err = [gCDevice newBufferWithBytes:&er0 length:sizeof(int)
		                                               options:MTLResourceStorageModeShared];

		if (!buf_z || !buf_x || !buf_y || !buf_b || !buf_w || !buf_err) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_iv];
		[enc setBuffer:buf_z offset:0 atIndex:0];
		[enc setBuffer:buf_x offset:0 atIndex:1];
		[enc setBuffer:buf_y offset:0 atIndex:2];
		[enc setBuffer:buf_b offset:0 atIndex:3];
		[enc setBuffer:buf_w offset:0 atIndex:4];
		[enc setBuffer:buf_err offset:0 atIndex:5];
		int t0 = T;
		int nz0 = nz;
		int nx0 = nx;
		int ny0 = ny;
		[enc setBytes:&t0 length:sizeof(t0) atIndex:6];
		[enc setBytes:&nz0 length:sizeof(nz0) atIndex:7];
		[enc setBytes:&nx0 length:sizeof(nx0) atIndex:8];
		[enc setBytes:&ny0 length:sizeof(ny0) atIndex:9];
		[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc endEncoding];

		rc = c_wait(cb);

		if (rc != 0) {
			return;
		}

		int ec = *((int *)[buf_err contents]);

		if (ec != 0) {
			rc = ec;
			return;
		}

		memcpy(beta_iv, [buf_b contents], (size_t)nx * (size_t)ny * sizeof(float));
	});

	return rc;
}

int metal_causal_cate(
	const float *X, const float *treatment, const float *Y,
	float *cate,
	int T, int nx) {
	if (c_check()) {
		return -3;
	}

	if (!X || !treatment || !Y || !cate || T <= 0 || nx <= 0) {
		return -1;
	}

	__block int rc = 0;

	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		int nfeat = nx + 1;
		unsigned long wf =
			(unsigned long)T * (unsigned long)nfeat + (unsigned long)T
			+ 3ul * (unsigned long)nfeat * (unsigned long)nfeat
			+ 5ul * (unsigned long)nfeat;

		id<MTLBuffer> buf_x = [gCDevice newBufferWithBytes:X
		                                           length:(size_t)T * (size_t)nx * sizeof(float)
		                                          options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_t = [gCDevice newBufferWithBytes:treatment length:(size_t)T * sizeof(float)
		                                            options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_y = [gCDevice newBufferWithBytes:Y length:(size_t)T * sizeof(float)
		                                           options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_c = [gCDevice newBufferWithLength:(size_t)T * sizeof(float)
		                                           options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_w = [gCDevice newBufferWithLength:(size_t)wf * sizeof(float)
		                                           options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_i = [gCDevice newBufferWithLength:(size_t)2 * (size_t)T * sizeof(int)
		                                           options:MTLResourceStorageModeShared];
		int           er0 = 0;
		id<MTLBuffer> buf_err = [gCDevice newBufferWithBytes:&er0 length:sizeof(int)
		                                               options:MTLResourceStorageModeShared];

		if (!buf_x || !buf_t || !buf_y || !buf_c || !buf_w || !buf_i || !buf_err) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_cate];
		[enc setBuffer:buf_x offset:0 atIndex:0];
		[enc setBuffer:buf_t offset:0 atIndex:1];
		[enc setBuffer:buf_y offset:0 atIndex:2];
		[enc setBuffer:buf_c offset:0 atIndex:3];
		[enc setBuffer:buf_w offset:0 atIndex:4];
		[enc setBuffer:buf_i offset:0 atIndex:5];
		int t0 = T;
		int nx0 = nx;
		[enc setBytes:&t0 length:sizeof(t0) atIndex:6];
		[enc setBytes:&nx0 length:sizeof(nx0) atIndex:7];
		[enc setBuffer:buf_err offset:0 atIndex:8];
		[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc endEncoding];

		rc = c_wait(cb);

		if (rc != 0) {
			return;
		}

		int ec = *((int *)[buf_err contents]);

		if (ec != 0) {
			rc = ec;
			return;
		}

		memcpy(cate, [buf_c contents], (size_t)T * sizeof(float));
	});

	return rc;
}

int metal_causal_dag_markov(
	const float *X, const float *adj,
	float *log_prob,
	int T, int N) {
	if (c_check()) {
		return -3;
	}

	if (!X || !adj || !log_prob || T <= 0 || N <= 0) {
		return -1;
	}

	__block int rc = 0;

	c_ensure_serial();
	dispatch_sync(gCSerial, ^{
		unsigned long wf =
			4ul * (unsigned long)T * (unsigned long)N
			+ 3ul * (unsigned long)T
			+ 4ul * (unsigned long)N * (unsigned long)N
			+ 8ul * (unsigned long)N + 64ul;
		unsigned long wi = 2ul * (unsigned long)N + (unsigned long)N * (unsigned long)N + (unsigned long)T;

		id<MTLBuffer> buf_x = [gCDevice newBufferWithBytes:X
		                                           length:(size_t)T * (size_t)N * sizeof(float)
		                                          options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_a = [gCDevice newBufferWithBytes:adj
		                                           length:(size_t)N * (size_t)N * sizeof(float)
		                                          options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_l = [gCDevice newBufferWithLength:(size_t)T * sizeof(float)
		                                           options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_w = [gCDevice newBufferWithLength:(size_t)wf * sizeof(float)
		                                           options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_i = [gCDevice newBufferWithLength:(size_t)wi * sizeof(int)
		                                           options:MTLResourceStorageModeShared];
		int           er0 = 0;
		id<MTLBuffer> buf_err = [gCDevice newBufferWithBytes:&er0 length:sizeof(int)
		                                               options:MTLResourceStorageModeShared];

		if (!buf_x || !buf_a || !buf_l || !buf_w || !buf_i || !buf_err) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gCQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_dag];
		[enc setBuffer:buf_x offset:0 atIndex:0];
		[enc setBuffer:buf_a offset:0 atIndex:1];
		[enc setBuffer:buf_l offset:0 atIndex:2];
		[enc setBuffer:buf_w offset:0 atIndex:3];
		[enc setBuffer:buf_i offset:0 atIndex:4];
		int t0 = T;
		int n0 = N;
		[enc setBytes:&t0 length:sizeof(t0) atIndex:5];
		[enc setBytes:&n0 length:sizeof(n0) atIndex:6];
		[enc setBuffer:buf_err offset:0 atIndex:7];
		[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc endEncoding];

		rc = c_wait(cb);

		if (rc != 0) {
			return;
		}

		int ec = *((int *)[buf_err contents]);

		if (ec != 0) {
			rc = ec;
			return;
		}

		memcpy(log_prob, [buf_l contents], (size_t)T * sizeof(float));
	});

	return rc;
}
