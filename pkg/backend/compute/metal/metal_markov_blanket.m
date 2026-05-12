#include <stddef.h>
#include <string.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "markov_blanket.h"

static id<MTLDevice> gMBDevice = nil;
static id<MTLCommandQueue> gMBQueue = nil;
static id<MTLComputePipelineState> gPSO_part = nil;
static id<MTLComputePipelineState> gPSO_fint = nil;
static id<MTLComputePipelineState> gPSO_fact = nil;

static id<MTLComputePipelineState> gPSO_mean = nil;
static id<MTLComputePipelineState> gPSO_cov = nil;
static id<MTLComputePipelineState> gPSO_joint = nil;
static id<MTLComputePipelineState> gPSO_chol = nil;

static int gMBInited = 0;
static dispatch_queue_t gMBSerial = NULL;

static void mb_ensure_serial(void) {
	static dispatch_once_t onceToken;
	dispatch_once(&onceToken, ^{
		gMBSerial = dispatch_queue_create("com.caramba.metal.markov_blanket", DISPATCH_QUEUE_SERIAL);
	});
}

static id<MTLComputePipelineState> mb_make_pso(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
	NSError *err = nil;
	id<MTLFunction> fn = [library newFunctionWithName:name];
	if (!fn) return nil;
	return [device newComputePipelineStateWithFunction:fn error:&err];
}

static int mb_wait(id<MTLCommandBuffer> cb) {
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

int metal_mb_init(const char *metallib_path) {
	if (!metallib_path || metallib_path[0] == '\0') return -1;
	mb_ensure_serial();
	__block int result = 0;
	dispatch_sync(gMBSerial, ^{
		if (gMBInited) return;
		gMBDevice = MTLCreateSystemDefaultDevice();
		if (!gMBDevice) { result = -1; return; }
		gMBQueue = [gMBDevice newCommandQueue];
		if (!gMBQueue) { result = -1; return; }
		NSString *path = [NSString stringWithUTF8String:metallib_path];
		NSError *err = nil;
		id<MTLLibrary> lib = [gMBDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
		if (!lib) { result = -1; return; }

		gPSO_part = mb_make_pso(gMBDevice, lib, @"mb_partition_kernel");
		gPSO_fint = mb_make_pso(gMBDevice, lib, @"mb_flow_internal_kernel");
		gPSO_fact = mb_make_pso(gMBDevice, lib, @"mb_flow_active_kernel");
		
		gPSO_mean = mb_make_pso(gMBDevice, lib, @"mb_mean_kernel");
		gPSO_cov = mb_make_pso(gMBDevice, lib, @"mb_cov_kernel");
		gPSO_joint = mb_make_pso(gMBDevice, lib, @"mb_joint_kernel");
		gPSO_chol = mb_make_pso(gMBDevice, lib, @"mb_chol_logdet_kernel");

		if (!gPSO_part || !gPSO_fint || !gPSO_fact || !gPSO_mean || !gPSO_cov || !gPSO_joint || !gPSO_chol) {
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
		gPSO_part = nil; gPSO_fint = nil; gPSO_fact = nil;
		gPSO_mean = nil; gPSO_cov = nil; gPSO_joint = nil; gPSO_chol = nil;
		gMBQueue = nil; gMBDevice = nil; gMBInited = 0;
	});
	return 0;
}

int metal_mb_partition(const float *x, const float *masks, float *out, int N, int Ns, int Na, int Ni, int Ne) {
	if (!gMBInited) return -3;
	if (!x || !masks || !out || N <= 0 || Ns < 0 || Na < 0 || Ni < 0 || Ne < 0) return -1;
	__block int rc = 0;
	mb_ensure_serial();
	dispatch_sync(gMBSerial, ^{
		size_t xb = (size_t)N * sizeof(float);
		size_t mb = (size_t)(4 * N) * sizeof(float);
		size_t outb = (size_t)(Ns + Na + Ni + Ne) * sizeof(float);
		id<MTLBuffer> buf_x = [gMBDevice newBufferWithBytes:x length:xb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_m = [gMBDevice newBufferWithBytes:masks length:mb options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf_o = [gMBDevice newBufferWithLength:outb options:MTLResourceStorageModeShared];
		int stzero = 0;
		id<MTLBuffer> buf_st = [gMBDevice newBufferWithBytes:&stzero length:sizeof(int) options:MTLResourceStorageModeShared];

		if (!buf_x || !buf_m || !buf_o || !buf_st) { rc = -4; return; }
		id<MTLCommandBuffer> cb = [gMBQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_part];
		[enc setBuffer:buf_x offset:0 atIndex:0];
		[enc setBuffer:buf_m offset:0 atIndex:1];
		[enc setBuffer:buf_o offset:0 atIndex:2];
		[enc setBuffer:buf_st offset:0 atIndex:3];
		[enc setBytes:&N length:sizeof(N) atIndex:4];
		[enc setBytes:&Ns length:sizeof(Ns) atIndex:5];
		[enc setBytes:&Na length:sizeof(Na) atIndex:6];
		[enc setBytes:&Ni length:sizeof(Ni) atIndex:7];
		[enc setBytes:&Ne length:sizeof(Ne) atIndex:8];
		[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc endEncoding];
		rc = mb_wait(cb);
		if (rc != 0) return;
		int st = *((int *)[buf_st contents]);
		if (st != 0) { rc = st; return; }
		memcpy(out, [buf_o contents], outb);
	});
	return rc;
}

int metal_mb_flow_internal(const float *x_sens, const float *W, const float *bias, float *out, int Ni, int Ns) {
	if (!gMBInited) return -3;
	if (!x_sens || !W || !bias || !out || Ni <= 0 || Ns <= 0) return -1;
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
		if (!buf_x || !buf_w || !buf_b || !buf_o) { rc = -4; return; }
		id<MTLCommandBuffer> cb = [gMBQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_fint];
		[enc setBuffer:buf_x offset:0 atIndex:0]; [enc setBuffer:buf_w offset:0 atIndex:1];
		[enc setBuffer:buf_b offset:0 atIndex:2]; [enc setBuffer:buf_o offset:0 atIndex:3];
		[enc setBytes:&Ni length:sizeof(Ni) atIndex:4]; [enc setBytes:&Ns length:sizeof(Ns) atIndex:5];
		[enc dispatchThreads:MTLSizeMake((NSUInteger)Ni, 1, 1) threadsPerThreadgroup:MTLSizeMake(gPSO_fint.threadExecutionWidth, 1, 1)];
		[enc endEncoding];
		rc = mb_wait(cb);
		if (rc == 0) memcpy(out, [buf_o contents], ob);
	});
	return rc;
}

int metal_mb_flow_active(const float *x_int, const float *W, const float *bias, float *out, int Na, int Ni) {
	if (!gMBInited) return -3;
	if (!x_int || !W || !bias || !out || Na <= 0 || Ni <= 0) return -1;
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
		if (!buf_x || !buf_w || !buf_b || !buf_o) { rc = -4; return; }
		id<MTLCommandBuffer> cb = [gMBQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_fact];
		[enc setBuffer:buf_x offset:0 atIndex:0]; [enc setBuffer:buf_w offset:0 atIndex:1];
		[enc setBuffer:buf_b offset:0 atIndex:2]; [enc setBuffer:buf_o offset:0 atIndex:3];
		[enc setBytes:&Na length:sizeof(Na) atIndex:4]; [enc setBytes:&Ni length:sizeof(Ni) atIndex:5];
		[enc dispatchThreads:MTLSizeMake((NSUInteger)Na, 1, 1) threadsPerThreadgroup:MTLSizeMake(gPSO_fact.threadExecutionWidth, 1, 1)];
		[enc endEncoding];
		rc = mb_wait(cb);
		if (rc == 0) memcpy(out, [buf_o contents], ob);
	});
	return rc;
}

int metal_mb_mutual_information(const float *X, const float *Y, float *out, int T, int N, int M) {
	if (!gMBInited) return -3;
	if (!X || !Y || !out || T < 2 || N <= 0 || M <= 0) return -1;
	__block int rc = 0;
	mb_ensure_serial();
	dispatch_sync(gMBSerial, ^{
		int NM = N + M;
		id<MTLBuffer> b_X = [gMBDevice newBufferWithBytes:X length:T*N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_Y = [gMBDevice newBufferWithBytes:Y length:T*M*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_out = [gMBDevice newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
		
		id<MTLBuffer> xm = [gMBDevice newBufferWithLength:N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> ym = [gMBDevice newBufferWithLength:M*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> cx = [gMBDevice newBufferWithLength:N*N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> cy = [gMBDevice newBufferWithLength:M*M*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> cxy = [gMBDevice newBufferWithLength:N*M*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> joint = [gMBDevice newBufferWithLength:NM*NM*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> work = [gMBDevice newBufferWithLength:NM*NM*sizeof(float) options:MTLResourceStorageModeShared];

		if (!b_X || !b_Y || !b_out || !xm || !ym || !cx || !cy || !cxy || !joint || !work) { rc = -4; return; }

		id<MTLCommandBuffer> cb = [gMBQueue commandBuffer];
		
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_mean];
		[enc setBuffer:b_X offset:0 atIndex:0]; [enc setBuffer:xm offset:0 atIndex:1];
		[enc setBytes:&T length:sizeof(T) atIndex:2]; [enc setBytes:&N length:sizeof(N) atIndex:3];
		[enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
		[enc endEncoding];

		enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_mean];
		[enc setBuffer:b_Y offset:0 atIndex:0]; [enc setBuffer:ym offset:0 atIndex:1];
		[enc setBytes:&T length:sizeof(T) atIndex:2]; [enc setBytes:&M length:sizeof(M) atIndex:3];
		[enc dispatchThreads:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
		[enc endEncoding];

		enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_cov];
		[enc setBuffer:b_X offset:0 atIndex:0]; [enc setBuffer:b_X offset:0 atIndex:1];
		[enc setBuffer:xm offset:0 atIndex:2]; [enc setBuffer:xm offset:0 atIndex:3];
		[enc setBuffer:cx offset:0 atIndex:4]; [enc setBytes:&T length:sizeof(T) atIndex:5];
		[enc setBytes:&N length:sizeof(N) atIndex:6]; [enc setBytes:&N length:sizeof(N) atIndex:7];
		[enc dispatchThreads:MTLSizeMake(N, N, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
		[enc endEncoding];

		enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_cov];
		[enc setBuffer:b_Y offset:0 atIndex:0]; [enc setBuffer:b_Y offset:0 atIndex:1];
		[enc setBuffer:ym offset:0 atIndex:2]; [enc setBuffer:ym offset:0 atIndex:3];
		[enc setBuffer:cy offset:0 atIndex:4]; [enc setBytes:&T length:sizeof(T) atIndex:5];
		[enc setBytes:&M length:sizeof(M) atIndex:6]; [enc setBytes:&M length:sizeof(M) atIndex:7];
		[enc dispatchThreads:MTLSizeMake(M, M, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
		[enc endEncoding];

		enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_cov];
		[enc setBuffer:b_X offset:0 atIndex:0]; [enc setBuffer:b_Y offset:0 atIndex:1];
		[enc setBuffer:xm offset:0 atIndex:2]; [enc setBuffer:ym offset:0 atIndex:3];
		[enc setBuffer:cxy offset:0 atIndex:4]; [enc setBytes:&T length:sizeof(T) atIndex:5];
		[enc setBytes:&N length:sizeof(N) atIndex:6]; [enc setBytes:&M length:sizeof(M) atIndex:7];
		[enc dispatchThreads:MTLSizeMake(N, M, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
		[enc endEncoding];

		enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_joint];
		[enc setBuffer:cx offset:0 atIndex:0]; [enc setBuffer:cy offset:0 atIndex:1];
		[enc setBuffer:cxy offset:0 atIndex:2]; [enc setBuffer:joint offset:0 atIndex:3];
		[enc setBytes:&N length:sizeof(N) atIndex:4]; [enc setBytes:&M length:sizeof(M) atIndex:5];
		[enc dispatchThreads:MTLSizeMake(NM, NM, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
		[enc endEncoding];

		enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_chol];
		[enc setBuffer:cx offset:0 atIndex:0]; [enc setBuffer:cy offset:0 atIndex:1];
		[enc setBuffer:joint offset:0 atIndex:2]; [enc setBuffer:b_out offset:0 atIndex:3];
		[enc setBuffer:work offset:0 atIndex:4]; [enc setBytes:&N length:sizeof(N) atIndex:5];
		[enc setBytes:&M length:sizeof(M) atIndex:6];
		[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc endEncoding];

		rc = mb_wait(cb);
		if (rc != 0) return;

		float v = ((float *)[b_out contents])[0];
		if (isnan(v)) { rc = -5; return; }
		out[0] = v;
	});
	return rc;
}
