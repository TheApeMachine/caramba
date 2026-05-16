#include <stddef.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "markov_blanket.h"

extern id<MTLDevice> gMBDevice;
extern id<MTLCommandQueue> gMBQueue;
extern id<MTLComputePipelineState> gPSO_part;
extern id<MTLComputePipelineState> gPSO_fint;
extern id<MTLComputePipelineState> gPSO_fact;
extern id<MTLComputePipelineState> gPSO_mean;
extern id<MTLComputePipelineState> gPSO_cov;
extern id<MTLComputePipelineState> gPSO_joint;
extern id<MTLComputePipelineState> gPSO_chol;
extern int gMBInited;
extern dispatch_queue_t gMBSerial;

void mb_ensure_serial(void);
int mb_wait(id<MTLCommandBuffer> cb);

int metal_mb_partition_tensor(
	const void *x, const void *masks, void *out,
	int N, int Ns, int Na, int Ni, int Ne) {
	if (!gMBInited) return -3;
	if (!x || !masks || !out || N <= 0 || Ns < 0 || Na < 0 || Ni < 0 || Ne < 0) return -1;

	__block int rc = 0;
	mb_ensure_serial();
	dispatch_sync(gMBSerial, ^{
		id<MTLBuffer> buf_x = (__bridge id)((void *)x);
		id<MTLBuffer> buf_m = (__bridge id)((void *)masks);
		id<MTLBuffer> buf_o = (__bridge id)out;
		int stzero = 0;
		id<MTLBuffer> buf_st = [gMBDevice newBufferWithBytes:&stzero
		                                               length:sizeof(int)
		                                              options:MTLResourceStorageModeShared];
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
		[enc dispatchThreads:MTLSizeMake(1, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc endEncoding];

		rc = mb_wait(cb);
		if (rc != 0) return;

		int st = *((int *)[buf_st contents]);
		if (st != 0) rc = st;
	});

	return rc;
}

int metal_mb_flow_internal_tensor(
	const void *x_sens, const void *W, const void *bias,
	void *out, int Ni, int Ns) {
	if (!gMBInited) return -3;
	if (!x_sens || !W || !bias || !out || Ni <= 0 || Ns <= 0) return -1;

	__block int rc = 0;
	mb_ensure_serial();
	dispatch_sync(gMBSerial, ^{
		id<MTLBuffer> buf_x = (__bridge id)((void *)x_sens);
		id<MTLBuffer> buf_w = (__bridge id)((void *)W);
		id<MTLBuffer> buf_b = (__bridge id)((void *)bias);
		id<MTLBuffer> buf_o = (__bridge id)out;
		if (!buf_x || !buf_w || !buf_b || !buf_o) { rc = -4; return; }

		id<MTLCommandBuffer> cb = [gMBQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_fint];
		[enc setBuffer:buf_x offset:0 atIndex:0];
		[enc setBuffer:buf_w offset:0 atIndex:1];
		[enc setBuffer:buf_b offset:0 atIndex:2];
		[enc setBuffer:buf_o offset:0 atIndex:3];
		[enc setBytes:&Ni length:sizeof(Ni) atIndex:4];
		[enc setBytes:&Ns length:sizeof(Ns) atIndex:5];
		[enc dispatchThreads:MTLSizeMake((NSUInteger)Ni, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(gPSO_fint.threadExecutionWidth, 1, 1)];
		[enc endEncoding];

		rc = mb_wait(cb);
	});

	return rc;
}

int metal_mb_flow_active_tensor(
	const void *x_int, const void *W, const void *bias,
	void *out, int Na, int Ni) {
	if (!gMBInited) return -3;
	if (!x_int || !W || !bias || !out || Na <= 0 || Ni <= 0) return -1;

	__block int rc = 0;
	mb_ensure_serial();
	dispatch_sync(gMBSerial, ^{
		id<MTLBuffer> buf_x = (__bridge id)((void *)x_int);
		id<MTLBuffer> buf_w = (__bridge id)((void *)W);
		id<MTLBuffer> buf_b = (__bridge id)((void *)bias);
		id<MTLBuffer> buf_o = (__bridge id)out;
		if (!buf_x || !buf_w || !buf_b || !buf_o) { rc = -4; return; }

		id<MTLCommandBuffer> cb = [gMBQueue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gPSO_fact];
		[enc setBuffer:buf_x offset:0 atIndex:0];
		[enc setBuffer:buf_w offset:0 atIndex:1];
		[enc setBuffer:buf_b offset:0 atIndex:2];
		[enc setBuffer:buf_o offset:0 atIndex:3];
		[enc setBytes:&Na length:sizeof(Na) atIndex:4];
		[enc setBytes:&Ni length:sizeof(Ni) atIndex:5];
		[enc dispatchThreads:MTLSizeMake((NSUInteger)Na, 1, 1)
		 threadsPerThreadgroup:MTLSizeMake(gPSO_fact.threadExecutionWidth, 1, 1)];
		[enc endEncoding];

		rc = mb_wait(cb);
	});

	return rc;
}

int metal_mb_mutual_information_tensor(
	const void *X, const void *Y, void *out,
	int T, int N, int M) {
	if (!gMBInited) return -3;
	if (!X || !Y || !out || T < 2 || N <= 0 || M <= 0) return -1;

	__block int rc = 0;
	mb_ensure_serial();
	dispatch_sync(gMBSerial, ^{
		int NM = N + M;
		id<MTLBuffer> b_X = (__bridge id)((void *)X);
		id<MTLBuffer> b_Y = (__bridge id)((void *)Y);
		id<MTLBuffer> b_out = (__bridge id)out;
		id<MTLBuffer> xm = [gMBDevice newBufferWithLength:N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> ym = [gMBDevice newBufferWithLength:M*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> cx = [gMBDevice newBufferWithLength:N*N*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> cy = [gMBDevice newBufferWithLength:M*M*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> cxy = [gMBDevice newBufferWithLength:N*M*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> joint = [gMBDevice newBufferWithLength:NM*NM*sizeof(float) options:MTLResourceStorageModeShared];
		id<MTLBuffer> work = [gMBDevice newBufferWithLength:NM*NM*sizeof(float) options:MTLResourceStorageModeShared];
		if (!b_X || !b_Y || !b_out || !xm || !ym || !cx || !cy || !cxy || !joint || !work) {
			rc = -4;
			return;
		}

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
	});

	return rc;
}
