#include <math.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stddef.h>
#include <string.h>
#include "vsa.h"

static id<MTLDevice>                gVDevice  = nil;
static id<MTLCommandQueue>          gVQueue   = nil;
static id<MTLComputePipelineState> gPSO_bind = nil;
static id<MTLComputePipelineState> gPSO_mul  = nil;
static id<MTLComputePipelineState> gPSO_scale = nil;
static id<MTLComputePipelineState> gPSO_sqr   = nil;
static id<MTLComputePipelineState> gPSO_red   = nil;
static id<MTLComputePipelineState> gPSO_fdot = nil;
static id<MTLComputePipelineState> gPSO_finv = nil;
static dispatch_queue_t             gVSerial = NULL;

static _Thread_local int  g_vsa_tls_code = 0;
static _Thread_local char g_vsa_tls_msg[256];

static void vsa_clear_tls(void) {
	g_vsa_tls_code = 0;
	g_vsa_tls_msg[0] = '\0';
}

static void vsa_fail(int code, const char *message) {
	g_vsa_tls_code = code;

	if (message == NULL) {
		g_vsa_tls_msg[0] = '\0';
		return;
	}

	strncpy(g_vsa_tls_msg, message, sizeof(g_vsa_tls_msg) - 1);
	g_vsa_tls_msg[sizeof(g_vsa_tls_msg) - 1] = '\0';
}

static void vsa_ensure_serial(void) {
	static dispatch_once_t onceToken;

	dispatch_once(&onceToken, ^{
		gVSerial = dispatch_queue_create("com.caramba.metal.vsa", DISPATCH_QUEUE_SERIAL);
	});
}

static id<MTLComputePipelineState> vsa_make_pso(
	id<MTLLibrary> library, NSString *name, NSError **out_error
) {
	NSError *pipeline_error = nil;
	id<MTLFunction> function = [library newFunctionWithName:name];

	if (!function) {
		if (out_error) {
			*out_error = [NSError errorWithDomain:@"caramba.vsa"
			                                  code:-1
			                              userInfo:@{
				NSLocalizedDescriptionKey : [NSString stringWithFormat:@"missing function %@", name],
			}];
		}

		return nil;
	}

	id<MTLComputePipelineState> pso =
		[gVDevice newComputePipelineStateWithFunction:function error:&pipeline_error];

	if (!pso && out_error) {
		*out_error = pipeline_error;
	}

	return pso;
}

static int vsa_wait_command_buffer(id<MTLCommandBuffer> command_buffer) {
	dispatch_semaphore_t done = dispatch_semaphore_create(0);

	if (!done) {
		return -1;
	}

	[command_buffer addCompletedHandler:^(id<MTLCommandBuffer> _) {
		dispatch_semaphore_signal(done);
	}];
	[command_buffer commit];
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);

	if (command_buffer.error != nil) {
		return -1;
	}

	return 0;
}

int metal_vsa_last_error_code(void) {
	return g_vsa_tls_code;
}

const char *metal_vsa_last_error_message(void) {
	return g_vsa_tls_msg;
}

int metal_vsa_init(const char *metallib_path) {
	vsa_clear_tls();
	vsa_ensure_serial();
	__block int result = 0;

	dispatch_sync(gVSerial, ^{
		if (metallib_path == NULL || metallib_path[0] == '\0') {
			vsa_fail(-1, "metallib_path empty");
			result = -1;
			return;
		}

		if (gVDevice != nil) {
			result = 0;
			return;
		}

		gVDevice = MTLCreateSystemDefaultDevice();

		if (!gVDevice) {
			vsa_fail(-1, "MTLCreateSystemDefaultDevice returned nil");
			result = -1;
			return;
		}

		gVQueue = [gVDevice newCommandQueue];

		if (!gVQueue) {
			vsa_fail(-1, "newCommandQueue failed");
			result = -1;
			return;
		}

		NSString *path         = [NSString stringWithUTF8String:metallib_path];
		NSError *library_error = nil;
		id<MTLLibrary> library = [gVDevice newLibraryWithURL:[NSURL fileURLWithPath:path]
		                                                  error:&library_error];

		if (!library) {
			vsa_fail(-1, "newLibraryWithURL failed");
			result = -1;
			return;
		}

		NSError *fe0 = nil;
		NSError *fe1 = nil;
		NSError *fe2 = nil;
		NSError *fe3 = nil;
		NSError *fe4 = nil;
		NSError *fe5 = nil;
		NSError *fe6 = nil;

		gPSO_bind  = vsa_make_pso(library, @"vsa_bind_kernel", &fe0);
		gPSO_mul   = vsa_make_pso(library, @"vsa_mul_kernel", &fe1);
		gPSO_scale = vsa_make_pso(library, @"vsa_scale_kernel", &fe2);
		gPSO_sqr   = vsa_make_pso(library, @"vsa_square_kernel", &fe3);
		gPSO_red   = vsa_make_pso(library, @"vsa_reduce_sum_atomic_kernel", &fe4);
		gPSO_fdot  = vsa_make_pso(library, @"vsa_finalize_dot_kernel", &fe5);
		gPSO_finv  = vsa_make_pso(library, @"vsa_finalize_invnorm_kernel", &fe6);

		if (!gPSO_bind || !gPSO_mul || !gPSO_scale || !gPSO_sqr || !gPSO_red || !gPSO_fdot ||
		    !gPSO_finv) {
			NSError *first = fe0 ? fe0 : (fe1 ? fe1 : (fe2 ? fe2 : (fe3 ? fe3 : (fe4 ? fe4 : (fe5 ? fe5 : fe6)))));
			const char *detail = first.localizedDescription.UTF8String;

			vsa_fail(-1, detail ? detail : "pipeline creation failed");
			gPSO_bind  = nil;
			gPSO_mul   = nil;
			gPSO_scale = nil;
			gPSO_sqr   = nil;
			gPSO_red   = nil;
			gPSO_fdot  = nil;
			gPSO_finv  = nil;
			gVQueue    = nil;
			gVDevice   = nil;
			result     = -1;
			return;
		}

		result = 0;
	});

	return result;
}

int metal_vsa_cleanup(void) {
	vsa_clear_tls();
	vsa_ensure_serial();
	dispatch_sync(gVSerial, ^{
		gPSO_bind  = nil;
		gPSO_mul   = nil;
		gPSO_scale = nil;
		gPSO_sqr   = nil;
		gPSO_red   = nil;
		gPSO_fdot  = nil;
		gPSO_finv  = nil;
		gVQueue    = nil;
		gVDevice   = nil;
	});

	return 0;
}

static id<MTLBuffer> vsa_buf_ro(const void *host, NSUInteger bytes) {
	return [gVDevice newBufferWithBytes:host length:bytes options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> vsa_buf_rw(NSUInteger bytes) {
	return [gVDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

int metal_vsa_bind(const float *a, const float *b, float *out, int n) {
	vsa_clear_tls();
	vsa_ensure_serial();
	__block int rc = 0;

	dispatch_sync(gVSerial, ^{
		if (!gVQueue || !gPSO_bind) {
			vsa_fail(-1, "VSA not initialised");
			rc = -1;
			return;
		}

		if (!a || !b || !out || n <= 0) {
			vsa_fail(-1, "invalid bind arguments");
			rc = -1;
			return;
		}

		NSUInteger            nb   = (NSUInteger)n * sizeof(float);
		id<MTLBuffer>           bufA = vsa_buf_ro(a, nb);
		id<MTLBuffer>           bufB = vsa_buf_ro(b, nb);
		id<MTLBuffer>           bufOut = vsa_buf_rw(nb);
		uint                    n_u  = (uint)n;
		id<MTLBuffer>           bufN = [gVDevice newBufferWithBytes:&n_u
		                                                      length:sizeof(uint)
		                                                     options:MTLResourceStorageModeShared];

		if (!bufA || !bufB || !bufOut || !bufN) {
			vsa_fail(-1, "buffer allocation failed");
			rc = -1;
			return;
		}

		id<MTLCommandBuffer>         cb = [gVQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

		[enc setComputePipelineState:gPSO_bind];
		[enc setBuffer:bufA   offset:0 atIndex:0];
		[enc setBuffer:bufB   offset:0 atIndex:1];
		[enc setBuffer:bufOut offset:0 atIndex:2];
		[enc setBuffer:bufN   offset:0 atIndex:3];

		NSUInteger tw = gPSO_bind.threadExecutionWidth;

		[enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
		[enc endEncoding];

		rc = vsa_wait_command_buffer(cb);

		if (rc != 0) {
			vsa_fail(-1, "command buffer failed (bind)");
			return;
		}

		memcpy(out, [bufOut contents], nb);
	});

	return rc;
}

int metal_vsa_l2normalize(const float *in, float *out, int n) {
	vsa_clear_tls();
	vsa_ensure_serial();
	__block int rc = 0;

	dispatch_sync(gVSerial, ^{
		if (!gVQueue || !gPSO_scale || !gPSO_sqr || !gPSO_red || !gPSO_finv) {
			vsa_fail(-1, "VSA not initialised");
			rc = -1;
			return;
		}

		if (!in || !out || n <= 0) {
			vsa_fail(-1, "invalid l2normalize arguments");
			rc = -1;
			return;
		}

		NSUInteger    nb      = (NSUInteger)n * sizeof(float);
		id<MTLBuffer> bufIn   = vsa_buf_ro(in, nb);
		id<MTLBuffer> bufSq   = vsa_buf_rw(nb);
		id<MTLBuffer> bufAt   = [gVDevice newBufferWithLength:sizeof(float)
		                                              options:MTLResourceStorageModeShared];
		id<MTLBuffer> bufInv  = [gVDevice newBufferWithLength:sizeof(float)
		                                               options:MTLResourceStorageModeShared];
		id<MTLBuffer> bufOut  = vsa_buf_rw(nb);
		uint          n_u     = (uint)n;
		id<MTLBuffer> bufN    = [gVDevice newBufferWithBytes:&n_u
		                                               length:sizeof(uint)
		                                              options:MTLResourceStorageModeShared];

		if (!bufIn || !bufSq || !bufAt || !bufInv || !bufOut || !bufN) {
			vsa_fail(-1, "buffer allocation failed");
			rc = -1;
			return;
		}

		id<MTLCommandBuffer> cb = [gVQueue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];

		[blit fillBuffer:bufAt range:NSMakeRange(0, sizeof(float)) value:0];
		[blit endEncoding];

		id<MTLComputeCommandEncoder> enc0 = [cb computeCommandEncoder];
		[enc0 setComputePipelineState:gPSO_sqr];
		[enc0 setBuffer:bufIn offset:0 atIndex:0];
		[enc0 setBuffer:bufSq offset:0 atIndex:1];
		[enc0 setBuffer:bufN  offset:0 atIndex:2];
		NSUInteger tw0 = gPSO_sqr.threadExecutionWidth;
		[enc0 dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(tw0, 1, 1)];
		[enc0 endEncoding];

		id<MTLComputeCommandEncoder> encR = [cb computeCommandEncoder];
		[encR setComputePipelineState:gPSO_red];
		[encR setBuffer:bufSq offset:0 atIndex:0];
		[encR setBuffer:bufAt offset:0 atIndex:1];
		[encR setBuffer:bufN  offset:0 atIndex:2];
		NSUInteger twR = gPSO_red.threadExecutionWidth;
		[encR dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(twR, 1, 1)];
		[encR endEncoding];

		id<MTLComputeCommandEncoder> encF = [cb computeCommandEncoder];
		[encF setComputePipelineState:gPSO_finv];
		[encF setBuffer:bufAt  offset:0 atIndex:0];
		[encF setBuffer:bufInv offset:0 atIndex:1];
		[encF dispatchThreads:MTLSizeMake(1, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[encF endEncoding];

		id<MTLComputeCommandEncoder> encS = [cb computeCommandEncoder];
		[encS setComputePipelineState:gPSO_scale];
		[encS setBuffer:bufIn   offset:0 atIndex:0];
		[encS setBuffer:bufOut  offset:0 atIndex:1];
		[encS setBuffer:bufInv  offset:0 atIndex:2];
		[encS setBuffer:bufN    offset:0 atIndex:3];
		NSUInteger twS = gPSO_scale.threadExecutionWidth;
		[encS dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(twS, 1, 1)];
		[encS endEncoding];

		rc = vsa_wait_command_buffer(cb);

		if (rc != 0) {
			vsa_fail(-1, "command buffer failed (l2normalize)");
			return;
		}

		memcpy(out, [bufOut contents], nb);
	});

	return rc;
}

int metal_vsa_dot(const float *a, const float *b, float *out, int n) {
	vsa_clear_tls();
	vsa_ensure_serial();
	__block int rc = 0;

	dispatch_sync(gVSerial, ^{
		if (!gVQueue || !gPSO_mul || !gPSO_red || !gPSO_fdot) {
			vsa_fail(-1, "VSA not initialised");
			rc = -1;
			return;
		}

		if (!a || !b || !out || n <= 0) {
			vsa_fail(-1, "invalid dot arguments");
			rc = -1;
			return;
		}

		NSUInteger    nb       = (NSUInteger)n * sizeof(float);
		id<MTLBuffer> bufA     = vsa_buf_ro(a, nb);
		id<MTLBuffer> bufB     = vsa_buf_ro(b, nb);
		id<MTLBuffer> bufProd  = vsa_buf_rw(nb);
		id<MTLBuffer> bufAt    = [gVDevice newBufferWithLength:sizeof(float)
		                                                options:MTLResourceStorageModeShared];
		id<MTLBuffer> bufDot   = [gVDevice newBufferWithLength:sizeof(float)
		                                                options:MTLResourceStorageModeShared];
		uint          n_u      = (uint)n;
		id<MTLBuffer> bufN     = [gVDevice newBufferWithBytes:&n_u
		                                               length:sizeof(uint)
		                                              options:MTLResourceStorageModeShared];

		if (!bufA || !bufB || !bufProd || !bufAt || !bufDot || !bufN) {
			vsa_fail(-1, "buffer allocation failed");
			rc = -1;
			return;
		}

		memset([bufAt contents], 0, sizeof(float));

		id<MTLCommandBuffer> cb = [gVQueue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];

		[blit fillBuffer:bufAt range:NSMakeRange(0, sizeof(float)) value:0];
		[blit endEncoding];

		id<MTLComputeCommandEncoder> encM = [cb computeCommandEncoder];
		[encM setComputePipelineState:gPSO_mul];
		[encM setBuffer:bufA    offset:0 atIndex:0];
		[encM setBuffer:bufB    offset:0 atIndex:1];
		[encM setBuffer:bufProd offset:0 atIndex:2];
		[encM setBuffer:bufN    offset:0 atIndex:3];
		NSUInteger twM = gPSO_mul.threadExecutionWidth;
		[encM dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(twM, 1, 1)];
		[encM endEncoding];

		id<MTLComputeCommandEncoder> encR = [cb computeCommandEncoder];
		[encR setComputePipelineState:gPSO_red];
		[encR setBuffer:bufProd offset:0 atIndex:0];
		[encR setBuffer:bufAt   offset:0 atIndex:1];
		[encR setBuffer:bufN    offset:0 atIndex:2];
		NSUInteger twR = gPSO_red.threadExecutionWidth;
		[encR dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(twR, 1, 1)];
		[encR endEncoding];

		id<MTLComputeCommandEncoder> encF = [cb computeCommandEncoder];
		[encF setComputePipelineState:gPSO_fdot];
		[encF setBuffer:bufAt  offset:0 atIndex:0];
		[encF setBuffer:bufDot offset:0 atIndex:1];
		[encF dispatchThreads:MTLSizeMake(1, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[encF endEncoding];

		rc = vsa_wait_command_buffer(cb);

		if (rc != 0) {
			vsa_fail(-1, "command buffer failed (dot)");
			return;
		}

		memcpy(out, [bufDot contents], sizeof(float));
	});

	return rc;
}
