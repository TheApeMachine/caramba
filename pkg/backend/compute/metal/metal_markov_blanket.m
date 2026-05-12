#include <math.h>
#include <stddef.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "markov_blanket.h"

static id<MTLDevice>                gMBDevice = nil;
static id<MTLCommandQueue>          gMBQueue  = nil;
static id<MTLComputePipelineState> gPSO_part = nil;
static id<MTLComputePipelineState> gPSO_fint = nil;
static id<MTLComputePipelineState> gPSO_fact = nil;
static id<MTLComputePipelineState> gPSO_mi   = nil;
static int                          gMBInited = 0;
static dispatch_queue_t             gMBSerial = NULL;

static void mb_ensure_serial(void) {
	static dispatch_once_t onceToken;

	dispatch_once(&onceToken, ^{
		gMBSerial = dispatch_queue_create("com.caramba.metal.markov_blanket", DISPATCH_QUEUE_SERIAL);
	});
}

static id<MTLComputePipelineState> mb_make_pso(
	id<MTLDevice> device, id<MTLLibrary> library, NSString *name
) {
	NSError *err = nil;
	id<MTLFunction> fn = [library newFunctionWithName:name];

	if (!fn) {
		return nil;
	}

	return [device newComputePipelineStateWithFunction:fn error:&err];
}

static int mb_wait(id<MTLCommandBuffer> cb) {
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

int metal_mb_init(const char *metallib_path) {
	if (metallib_path == NULL || metallib_path[0] == '\0') {
		return -1;
	}

	mb_ensure_serial();
	__block int result = 0;

	dispatch_sync(gMBSerial, ^{
		if (gMBInited) {
			return;
		}

		gMBDevice = MTLCreateSystemDefaultDevice();

		if (!gMBDevice) {
			result = -1;
			return;
		}

		gMBQueue = [gMBDevice newCommandQueue];

		if (!gMBQueue) {
			result = -1;
			return;
		}

		NSString *path = [NSString stringWithUTF8String:metallib_path];
		NSError *err   = nil;
		id<MTLLibrary> lib = [gMBDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];

		if (!lib) {
			result = -1;
			return;
		}

		gPSO_part = mb_make_pso(gMBDevice, lib, @"mb_partition_kernel");
		gPSO_fint = mb_make_pso(gMBDevice, lib, @"mb_flow_internal_kernel");
		gPSO_fact = mb_make_pso(gMBDevice, lib, @"mb_flow_active_kernel");
		gPSO_mi   = mb_make_pso(gMBDevice, lib, @"mb_mutual_information_kernel");

		if (!gPSO_part || !gPSO_fint || !gPSO_fact || !gPSO_mi) {
			result = -1;
			return;
		}

		gMBInited = 1;
	});

	return result;
}

int metal_mb_cleanup(void) {
	mb_ensure_serial();
	dispatch_sync(gMBSerial, ^{
		gPSO_part = nil;
		gPSO_fint = nil;
		gPSO_fact = nil;
		gPSO_mi   = nil;
		gMBQueue  = nil;
		gMBDevice = nil;
		gMBInited = 0;
	});

	return 0;
}

int metal_mb_partition(
	const float *x, const float *masks,
	float *out,
	int N, int Ns, int Na, int Ni, int Ne) {
	if (!gMBInited) {
		return -3;
	}

	if (!x || !masks || !out || N <= 0 || Ns < 0 || Na < 0 || Ni < 0 || Ne < 0) {
		return -1;
	}

	__block int rc = 0;

	mb_ensure_serial();
	dispatch_sync(gMBSerial, ^{
		size_t xb     = (size_t)N * sizeof(float);
		size_t mb     = (size_t)(4 * N) * sizeof(float);
		size_t outb   = (size_t)(Ns + Na + Ni + Ne) * sizeof(float);
		id<MTLBuffer> buf_x  = [gMBDevice newBufferWithBytes:x length:xb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_m  = [gMBDevice newBufferWithBytes:masks length:mb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_o  = [gMBDevice newBufferWithLength:outb options:MTLResourceStorageModeShared];
		int           stzero = 0;
		id<MTLBuffer> buf_st = [gMBDevice newBufferWithBytes:&stzero length:sizeof(int) options:MTLResourceStorageModeShared];

		if (!buf_x || !buf_m || !buf_o || !buf_st) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gMBQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_part];
		[enc setBuffer:buf_x offset:0 atIndex:0];
		[enc setBuffer:buf_m offset:0 atIndex:1];
		[enc setBuffer:buf_o offset:0 atIndex:2];
		[enc setBuffer:buf_st offset:0 atIndex:3];
		int n0 = N, ns0 = Ns, na0 = Na, ni0 = Ni, ne0 = Ne;
		[enc setBytes:&n0 length:sizeof(n0) atIndex:4];
		[enc setBytes:&ns0 length:sizeof(ns0) atIndex:5];
		[enc setBytes:&na0 length:sizeof(na0) atIndex:6];
		[enc setBytes:&ni0 length:sizeof(ni0) atIndex:7];
		[enc setBytes:&ne0 length:sizeof(ne0) atIndex:8];
		[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc endEncoding];

		rc = mb_wait(cb);

		if (rc != 0) {
			return;
		}

		int st = *((int *)[buf_st contents]);

		if (st != 0) {
			rc = st;
			return;
		}

		memcpy(out, [buf_o contents], outb);
	});

	return rc;
}

int metal_mb_flow_internal(
	const float *x_sens, const float *W, const float *bias,
	float *out,
	int Ni, int Ns) {
	if (!gMBInited) {
		return -3;
	}

	if (!x_sens || !W || !bias || !out || Ni <= 0 || Ns <= 0) {
		return -1;
	}

	__block int rc = 0;

	mb_ensure_serial();
	dispatch_sync(gMBSerial, ^{
		size_t sb = (size_t)Ns * sizeof(float);
		size_t wb = (size_t)Ni * (size_t)Ns * sizeof(float);
		size_t bb = (size_t)Ni * sizeof(float);
		size_t ob = (size_t)Ni * sizeof(float);

		id<MTLBuffer> buf_x = [gMBDevice newBufferWithBytes:x_sens length:sb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_w = [gMBDevice newBufferWithBytes:W length:wb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_b = [gMBDevice newBufferWithBytes:bias length:bb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_o = [gMBDevice newBufferWithLength:ob options:MTLResourceStorageModeShared];

		if (!buf_x || !buf_w || !buf_b || !buf_o) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gMBQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_fint];
		[enc setBuffer:buf_x offset:0 atIndex:0];
		[enc setBuffer:buf_w offset:0 atIndex:1];
		[enc setBuffer:buf_b offset:0 atIndex:2];
		[enc setBuffer:buf_o offset:0 atIndex:3];
		int ni0 = Ni, ns0 = Ns;
		[enc setBytes:&ni0 length:sizeof(ni0) atIndex:4];
		[enc setBytes:&ns0 length:sizeof(ns0) atIndex:5];

		NSUInteger tw = gPSO_fint.threadExecutionWidth;

		[enc dispatchThreads:MTLSizeMake((NSUInteger)Ni, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
		[enc endEncoding];

		rc = mb_wait(cb);

		if (rc != 0) {
			return;
		}

		memcpy(out, [buf_o contents], ob);
	});

	return rc;
}

int metal_mb_flow_active(
	const float *x_int, const float *W, const float *bias,
	float *out,
	int Na, int Ni) {
	if (!gMBInited) {
		return -3;
	}

	if (!x_int || !W || !bias || !out || Na <= 0 || Ni <= 0) {
		return -1;
	}

	__block int rc = 0;

	mb_ensure_serial();
	dispatch_sync(gMBSerial, ^{
		size_t xb = (size_t)Ni * sizeof(float);
		size_t wb = (size_t)Na * (size_t)Ni * sizeof(float);
		size_t bb = (size_t)Na * sizeof(float);
		size_t ob = (size_t)Na * sizeof(float);

		id<MTLBuffer> buf_x = [gMBDevice newBufferWithBytes:x_int length:xb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_w = [gMBDevice newBufferWithBytes:W length:wb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_b = [gMBDevice newBufferWithBytes:bias length:bb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_o = [gMBDevice newBufferWithLength:ob options:MTLResourceStorageModeShared];

		if (!buf_x || !buf_w || !buf_b || !buf_o) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gMBQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_fact];
		[enc setBuffer:buf_x offset:0 atIndex:0];
		[enc setBuffer:buf_w offset:0 atIndex:1];
		[enc setBuffer:buf_b offset:0 atIndex:2];
		[enc setBuffer:buf_o offset:0 atIndex:3];
		int na0 = Na, ni0 = Ni;
		[enc setBytes:&na0 length:sizeof(na0) atIndex:4];
		[enc setBytes:&ni0 length:sizeof(ni0) atIndex:5];

		NSUInteger tw = gPSO_fact.threadExecutionWidth;

		[enc dispatchThreads:MTLSizeMake((NSUInteger)Na, 1, 1)
		  threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
		[enc endEncoding];

		rc = mb_wait(cb);

		if (rc != 0) {
			return;
		}

		memcpy(out, [buf_o contents], ob);
	});

	return rc;
}

int metal_mb_mutual_information(
	const float *X, const float *Y,
	float *out,
	int T, int N, int M) {
	if (!gMBInited) {
		return -3;
	}

	if (!X || !Y || !out || T < 2 || N <= 0 || M <= 0) {
		return -1;
	}

	__block int rc = 0;

	mb_ensure_serial();
	dispatch_sync(gMBSerial, ^{
		int             nm      = N + M;
		size_t          scratchElems =
			(size_t)N + (size_t)M + (size_t)N * (size_t)N + (size_t)M * (size_t)M +
			(size_t)N * (size_t)M + 2 * (size_t)nm * (size_t)nm;
		size_t          xb = (size_t)T * (size_t)N * sizeof(float);
		size_t          yb = (size_t)T * (size_t)M * sizeof(float);
		id<MTLBuffer> buf_x = [gMBDevice newBufferWithBytes:X length:xb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_y = [gMBDevice newBufferWithBytes:Y length:yb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_o = [gMBDevice newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_s = [gMBDevice newBufferWithLength:scratchElems * sizeof(float)
		                                             options:MTLResourceStorageModeShared];

		if (!buf_x || !buf_y || !buf_o || !buf_s) {
			rc = -4;
			return;
		}

		id<MTLCommandBuffer>         cb  = [gMBQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_mi];
		[enc setBuffer:buf_x offset:0 atIndex:0];
		[enc setBuffer:buf_y offset:0 atIndex:1];
		[enc setBuffer:buf_o offset:0 atIndex:2];
		[enc setBuffer:buf_s offset:0 atIndex:3];
		int t0 = T, n0 = N, m0 = M;
		[enc setBytes:&t0 length:sizeof(t0) atIndex:4];
		[enc setBytes:&n0 length:sizeof(n0) atIndex:5];
		[enc setBytes:&m0 length:sizeof(m0) atIndex:6];
		[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc endEncoding];

		rc = mb_wait(cb);

		if (rc != 0) {
			return;
		}

		float v = ((float *)[buf_o contents])[0];

		if (isnan(v)) {
			rc = -5;
			return;
		}

		out[0] = v;
	});

	return rc;
}
