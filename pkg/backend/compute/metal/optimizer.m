#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "optimizer.h"

static id<MTLDevice> gOptimizerDevice = nil;
static id<MTLCommandQueue> gOptimizerQueue = nil;
static id<MTLLibrary> gOptimizerLibrary = nil;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *gOptimizerPipelines = nil;
static NSString *gOptimizerLibraryPath = nil;
static char gOptimizerLastError[1024] = {0};

static int optimizer_error(int code, NSString *message) {
	if (message) {
		snprintf(gOptimizerLastError, sizeof(gOptimizerLastError), "%s", [message UTF8String]);
	} else {
		snprintf(gOptimizerLastError, sizeof(gOptimizerLastError), "metal optimizer error %d", code);
	}
	return code;
}

const char *metal_optimizer_last_error(void) {
	return gOptimizerLastError;
}

int metal_optimizer_init(const char *metallib_path) {
	@autoreleasepool {
		if (!metallib_path || metallib_path[0] == '\0') {
			return optimizer_error(-1, @"empty optimizer metallib path");
		}
		if (gOptimizerLibrary != nil) return 0;
		gOptimizerLibraryPath = [[NSString alloc] initWithUTF8String:metallib_path];
		if (!gOptimizerLibraryPath) return optimizer_error(-1, @"invalid optimizer metallib path");
		gOptimizerDevice = MTLCreateSystemDefaultDevice();
		if (!gOptimizerDevice) return optimizer_error(-1, @"no Metal device");
		gOptimizerQueue = [gOptimizerDevice newCommandQueue];
		if (!gOptimizerQueue) return optimizer_error(-1, @"could not create Metal command queue");
		NSError *error = nil;
		NSURL *url = [NSURL fileURLWithPath:gOptimizerLibraryPath];
		gOptimizerLibrary = [gOptimizerDevice newLibraryWithURL:url error:&error];
		if (!gOptimizerLibrary) return optimizer_error(-1, [error localizedDescription]);
		gOptimizerPipelines = [[NSMutableDictionary alloc] init];
		gOptimizerLastError[0] = '\0';
		return 0;
	}
}

static int optimizer_init(void) {
	@autoreleasepool {
		if (gOptimizerDevice && gOptimizerQueue && gOptimizerLibrary) return 0;
		return optimizer_error(-1, @"optimizer metallib not initialized");
	}
}

static id<MTLComputePipelineState> optimizer_pipeline(const char *name) {
	NSString *key = [NSString stringWithUTF8String:name];
	id<MTLComputePipelineState> cached = [gOptimizerPipelines objectForKey:key];
	if (cached) return cached;
	id<MTLFunction> function = [gOptimizerLibrary newFunctionWithName:key];
	if (!function) optimizer_error(-2, [NSString stringWithFormat:@"missing Metal function %@", key]);
	if (!function) return nil;
	NSError *error = nil;
	id<MTLComputePipelineState> pipeline = [gOptimizerDevice newComputePipelineStateWithFunction:function error:&error];
	if (!pipeline) optimizer_error(-2, [error localizedDescription]);
	if (!pipeline) return nil;
	[gOptimizerPipelines setObject:pipeline forKey:key];
	return pipeline;
}

static float *to_float(const double *values, int count) {
	float *converted = (float *)malloc((size_t)count * sizeof(float));
	if (!converted) return NULL;
	for (int index = 0; index < count; index++) converted[index] = (float)values[index];
	return converted;
}

static void to_double(double *out, const float *values, int count) {
	for (int index = 0; index < count; index++) out[index] = (double)values[index];
}

static id<MTLBuffer> ro(const float *values, int count) {
	return [gOptimizerDevice newBufferWithBytes:values length:(NSUInteger)count * sizeof(float) options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> rw(const float *values, int count) {
	id<MTLBuffer> buffer = [gOptimizerDevice newBufferWithLength:(NSUInteger)count * sizeof(float) options:MTLResourceStorageModeShared];
	if (values && count > 0) memcpy([buffer contents], values, (size_t)count * sizeof(float));
	return buffer;
}

static id<MTLBuffer> rw_uint(const uint *values, int count) {
	id<MTLBuffer> buffer = [gOptimizerDevice newBufferWithLength:(NSUInteger)count * sizeof(uint) options:MTLResourceStorageModeShared];
	if (values && count > 0) memcpy([buffer contents], values, (size_t)count * sizeof(uint));
	return buffer;
}

static int run_threads(id<MTLComputePipelineState> pipeline, id<MTLCommandBuffer> command_buffer, id<MTLComputeCommandEncoder> encoder, int count) {
	if (!pipeline || !command_buffer || !encoder) return -1;
	NSUInteger thread_count = (NSUInteger)count;
	NSUInteger group_count = thread_count < 256 ? thread_count : 256;
	if (group_count == 0) return optimizer_error(-1, @"empty Metal dispatch");
	[encoder dispatchThreads:MTLSizeMake(thread_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(group_count, 1, 1)];
	[encoder endEncoding];
	[command_buffer commit];
	[command_buffer waitUntilCompleted];
	return command_buffer.error ? optimizer_error(-1, [command_buffer.error localizedDescription]) : 0;
}

static int optimizer_groups(int count) {
	return count <= 0 ? 0 : (count + 255) / 256;
}

	#define START_ENCODER(kernel_name) \
		if (optimizer_init() != 0) return -1; \
		id<MTLComputePipelineState> pipeline = optimizer_pipeline(kernel_name); \
		if (!pipeline) return -2; \
		id<MTLCommandBuffer> command_buffer = [gOptimizerQueue commandBuffer]; \
		if (!command_buffer) return -3; \
		id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder]; \
		if (!encoder) return -4; \
		[encoder setComputePipelineState:pipeline]

#define SET_SCALAR(value, type_name, index) do { type_name scalar = (type_name)(value); [encoder setBytes:&scalar length:sizeof(type_name) atIndex:index]; } while (0)

static int reduce_pair_partials(id<MTLBuffer> partials, int count, id<MTLBuffer> *reduced) {
	id<MTLBuffer> current = partials;
	int current_count = count;

	while (current_count > 1) {
		int next_count = optimizer_groups(current_count);
		id<MTLBuffer> next = rw(NULL, next_count * 2);
		int rc = 0;

		{
			START_ENCODER("reduce_pair_sums");
			[encoder setBuffer:current offset:0 atIndex:0];
			[encoder setBuffer:next offset:0 atIndex:1];
			SET_SCALAR(current_count, uint, 2);
			rc = run_threads(pipeline, command_buffer, encoder, next_count * 256);
		}

		if (rc != 0) return rc;
		current = next;
		current_count = next_count;
	}

	*reduced = current;
	return 0;
}

static int finish_state(double *out, id<MTLBuffer> out_buffer, double *first, id<MTLBuffer> first_buffer, double *second, id<MTLBuffer> second_buffer, int count) {
	to_double(out, (const float *)[out_buffer contents], count);
	if (first && first_buffer) to_double(first, (const float *)[first_buffer contents], count);
	if (second && second_buffer) to_double(second, (const float *)[second_buffer contents], count);
	return 0;
}

int metal_optimizer_adam(double *out, double *moment, double *variance, const double *params, const double *grads, int count, double beta1, double beta2, double learning_rate, double eps) {
	float *m = to_float(moment, count), *v = to_float(variance, count), *p = to_float(params, count), *g = to_float(grads, count);
	START_ENCODER("adam");
	id<MTLBuffer> bo = rw(NULL, count), bm = rw(m, count), bv = rw(v, count), bp = ro(p, count), bg = ro(g, count);
	[encoder setBuffer:bo offset:0 atIndex:0]; [encoder setBuffer:bm offset:0 atIndex:1]; [encoder setBuffer:bv offset:0 atIndex:2]; [encoder setBuffer:bp offset:0 atIndex:3]; [encoder setBuffer:bg offset:0 atIndex:4];
	SET_SCALAR(count, uint, 5); SET_SCALAR(beta1, float, 6); SET_SCALAR(beta2, float, 7); SET_SCALAR(learning_rate, float, 8); SET_SCALAR(eps, float, 9);
	int rc = run_threads(pipeline, command_buffer, encoder, count);
	if (rc == 0) finish_state(out, bo, moment, bm, variance, bv, count);
	free(m); free(v); free(p); free(g); return rc;
}

int metal_optimizer_adamw(double *out, double *moment, double *variance, const double *params, const double *grads, int count, double beta1, double beta2, double learning_rate, double eps, double weight_decay_step) {
	float *m = to_float(moment, count), *v = to_float(variance, count), *p = to_float(params, count), *g = to_float(grads, count);
	START_ENCODER("adamw");
	id<MTLBuffer> bo = rw(NULL, count), bm = rw(m, count), bv = rw(v, count), bp = ro(p, count), bg = ro(g, count);
	[encoder setBuffer:bo offset:0 atIndex:0]; [encoder setBuffer:bm offset:0 atIndex:1]; [encoder setBuffer:bv offset:0 atIndex:2]; [encoder setBuffer:bp offset:0 atIndex:3]; [encoder setBuffer:bg offset:0 atIndex:4];
	SET_SCALAR(count, uint, 5); SET_SCALAR(beta1, float, 6); SET_SCALAR(beta2, float, 7); SET_SCALAR(learning_rate, float, 8); SET_SCALAR(eps, float, 9); SET_SCALAR(weight_decay_step, float, 10);
	int rc = run_threads(pipeline, command_buffer, encoder, count);
	if (rc == 0) finish_state(out, bo, moment, bm, variance, bv, count);
	free(m); free(v); free(p); free(g); return rc;
}

int metal_optimizer_adamax(double *out, double *moment, double *infinity_norm, const double *params, const double *grads, int count, double beta1, double beta2, double learning_rate, double eps) {
	float *m = to_float(moment, count), *u = to_float(infinity_norm, count), *p = to_float(params, count), *g = to_float(grads, count);
	START_ENCODER("adamax");
	id<MTLBuffer> bo = rw(NULL, count), bm = rw(m, count), bu = rw(u, count), bp = ro(p, count), bg = ro(g, count);
	[encoder setBuffer:bo offset:0 atIndex:0]; [encoder setBuffer:bm offset:0 atIndex:1]; [encoder setBuffer:bu offset:0 atIndex:2]; [encoder setBuffer:bp offset:0 atIndex:3]; [encoder setBuffer:bg offset:0 atIndex:4];
	SET_SCALAR(count, uint, 5); SET_SCALAR(beta1, float, 6); SET_SCALAR(beta2, float, 7); SET_SCALAR(learning_rate, float, 8); SET_SCALAR(eps, float, 9);
	int rc = run_threads(pipeline, command_buffer, encoder, count);
	if (rc == 0) finish_state(out, bo, moment, bm, infinity_norm, bu, count);
	free(m); free(u); free(p); free(g); return rc;
}

int metal_optimizer_sgd(double *out, double *velocity, const double *params, const double *grads, int count, double learning_rate, double weight_decay, double momentum, int nesterov) {
	float *vel = to_float(velocity, count), *p = to_float(params, count), *g = to_float(grads, count);
	START_ENCODER("sgd");
	id<MTLBuffer> bo = rw(NULL, count), bv = rw(vel, count), bp = ro(p, count), bg = ro(g, count);
	[encoder setBuffer:bo offset:0 atIndex:0]; [encoder setBuffer:bv offset:0 atIndex:1]; [encoder setBuffer:bp offset:0 atIndex:2]; [encoder setBuffer:bg offset:0 atIndex:3];
	SET_SCALAR(count, uint, 4); SET_SCALAR(learning_rate, float, 5); SET_SCALAR(weight_decay, float, 6); SET_SCALAR(momentum, float, 7); SET_SCALAR(nesterov, uint, 8);
	int rc = run_threads(pipeline, command_buffer, encoder, count);
	if (rc == 0) finish_state(out, bo, velocity, bv, NULL, nil, count);
	free(vel); free(p); free(g); return rc;
}

int metal_optimizer_lion(double *out, double *moment, const double *params, const double *grads, int count, double learning_rate, double beta1, double beta2, double weight_decay) {
	float *m = to_float(moment, count), *p = to_float(params, count), *g = to_float(grads, count);
	START_ENCODER("lion");
	id<MTLBuffer> bo = rw(NULL, count), bm = rw(m, count), bp = ro(p, count), bg = ro(g, count);
	[encoder setBuffer:bo offset:0 atIndex:0]; [encoder setBuffer:bm offset:0 atIndex:1]; [encoder setBuffer:bp offset:0 atIndex:2]; [encoder setBuffer:bg offset:0 atIndex:3];
	SET_SCALAR(count, uint, 4); SET_SCALAR(learning_rate, float, 5); SET_SCALAR(beta1, float, 6); SET_SCALAR(beta2, float, 7); SET_SCALAR(weight_decay, float, 8);
	int rc = run_threads(pipeline, command_buffer, encoder, count);
	if (rc == 0) finish_state(out, bo, moment, bm, NULL, nil, count);
	free(m); free(p); free(g); return rc;
}

int metal_optimizer_rmsprop(double *out, double *square_average, double *momentum_buffer, double *grad_average, const double *params, const double *grads, int count, double learning_rate, double alpha, double eps, double momentum, double weight_decay, int centered) {
	float *sq = to_float(square_average, count), *buf = to_float(momentum_buffer, count), *ga = to_float(grad_average, count), *p = to_float(params, count), *g = to_float(grads, count);
	START_ENCODER("rmsprop");
	id<MTLBuffer> bo = rw(NULL, count), bsq = rw(sq, count), bbuf = rw(buf, count), bga = rw(ga, count), bp = ro(p, count), bg = ro(g, count);
	[encoder setBuffer:bo offset:0 atIndex:0]; [encoder setBuffer:bsq offset:0 atIndex:1]; [encoder setBuffer:bbuf offset:0 atIndex:2]; [encoder setBuffer:bga offset:0 atIndex:3]; [encoder setBuffer:bp offset:0 atIndex:4]; [encoder setBuffer:bg offset:0 atIndex:5];
	SET_SCALAR(count, uint, 6); SET_SCALAR(learning_rate, float, 7); SET_SCALAR(alpha, float, 8); SET_SCALAR(eps, float, 9); SET_SCALAR(momentum, float, 10); SET_SCALAR(weight_decay, float, 11); SET_SCALAR(centered, uint, 12);
	int rc = run_threads(pipeline, command_buffer, encoder, count);
	if (rc == 0) { to_double(out, [bo contents], count); to_double(square_average, [bsq contents], count); to_double(momentum_buffer, [bbuf contents], count); to_double(grad_average, [bga contents], count); }
	free(sq); free(buf); free(ga); free(p); free(g); return rc;
}

int metal_optimizer_hebbian(double *out, const double *params, const double *grads, int count, double learning_rate, double max_norm) {
	float *p = to_float(params, count), *g = to_float(grads, count);
	START_ENCODER("hebbian");
	id<MTLBuffer> bo = rw(NULL, count), bn = rw(NULL, 1), bp = ro(p, count), bg = ro(g, count);
	float zero = 0; memcpy([bn contents], &zero, sizeof(float));
	[encoder setBuffer:bo offset:0 atIndex:0]; [encoder setBuffer:bn offset:0 atIndex:1]; [encoder setBuffer:bp offset:0 atIndex:2]; [encoder setBuffer:bg offset:0 atIndex:3];
	SET_SCALAR(count, uint, 4); SET_SCALAR(learning_rate, float, 5);
	int rc = run_threads(pipeline, command_buffer, encoder, count);
	if (rc == 0 && max_norm > 0) {
		float norm_square = *((float *)[bn contents]);
		float norm = sqrtf(norm_square);
		if (norm > max_norm) {
			START_ENCODER("scale");
			[encoder setBuffer:bo offset:0 atIndex:0]; SET_SCALAR(count, uint, 1); SET_SCALAR(((float)max_norm / norm), float, 2);
			rc = run_threads(pipeline, command_buffer, encoder, count);
		}
	}
	if (rc == 0) to_double(out, [bo contents], count);
	free(p); free(g); return rc;
}

int metal_optimizer_lars(double *out, double *velocity, const double *params, const double *grads, int count, double learning_rate, double eta, double momentum, double weight_decay, double eps) {
	float *v = to_float(velocity, count), *p = to_float(params, count), *g = to_float(grads, count);
	id<MTLBuffer> bo = rw(NULL, count), bv = rw(v, count), bp = ro(p, count), bg = ro(g, count);
	int group_count = optimizer_groups(count);
	id<MTLBuffer> partials = rw(NULL, group_count * 2), norms = nil;
	int rc = 0;

	{
		START_ENCODER("lars_norms");
		[encoder setBuffer:bp offset:0 atIndex:0]; [encoder setBuffer:bg offset:0 atIndex:1]; [encoder setBuffer:partials offset:0 atIndex:2];
		SET_SCALAR(count, uint, 3);
		rc = run_threads(pipeline, command_buffer, encoder, group_count * 256);
	}

	if (rc == 0) rc = reduce_pair_partials(partials, group_count, &norms);

	if (rc == 0) {
		START_ENCODER("lars_apply");
		[encoder setBuffer:bo offset:0 atIndex:0]; [encoder setBuffer:bv offset:0 atIndex:1]; [encoder setBuffer:bp offset:0 atIndex:2]; [encoder setBuffer:bg offset:0 atIndex:3]; [encoder setBuffer:norms offset:0 atIndex:4];
		SET_SCALAR(count, uint, 5); SET_SCALAR(learning_rate, float, 6); SET_SCALAR(eta, float, 7); SET_SCALAR(momentum, float, 8); SET_SCALAR(weight_decay, float, 9); SET_SCALAR(eps, float, 10);
		rc = run_threads(pipeline, command_buffer, encoder, count);
	}

	if (rc == 0) finish_state(out, bo, velocity, bv, NULL, nil, count);
	free(v); free(p); free(g); return rc;
}

int metal_optimizer_lamb(double *out, double *moment, double *variance, const double *params, const double *grads, int count, double learning_rate, double beta1, double beta2, double eps, double weight_decay, double bias_correction1_inv, double bias_correction2_inv) {
	float *m = to_float(moment, count), *v = to_float(variance, count), *p = to_float(params, count), *g = to_float(grads, count);
	id<MTLBuffer> bo = rw(NULL, count), bm = rw(m, count), bv = rw(v, count), bp = ro(p, count), bg = ro(g, count);
	int group_count = optimizer_groups(count);
	id<MTLBuffer> partials = rw(NULL, group_count * 2), norms = nil;
	int rc = 0;

	{
		START_ENCODER("lamb_prepare");
		[encoder setBuffer:bo offset:0 atIndex:0]; [encoder setBuffer:bm offset:0 atIndex:1]; [encoder setBuffer:bv offset:0 atIndex:2]; [encoder setBuffer:bp offset:0 atIndex:3]; [encoder setBuffer:bg offset:0 atIndex:4]; [encoder setBuffer:partials offset:0 atIndex:5];
		SET_SCALAR(count, uint, 6); SET_SCALAR(beta1, float, 7); SET_SCALAR(beta2, float, 8); SET_SCALAR(eps, float, 9); SET_SCALAR(weight_decay, float, 10); SET_SCALAR(bias_correction1_inv, float, 11); SET_SCALAR(bias_correction2_inv, float, 12);
		rc = run_threads(pipeline, command_buffer, encoder, group_count * 256);
	}

	if (rc == 0) rc = reduce_pair_partials(partials, group_count, &norms);

	if (rc == 0) {
		START_ENCODER("lamb_apply");
		[encoder setBuffer:bo offset:0 atIndex:0]; [encoder setBuffer:bo offset:0 atIndex:1]; [encoder setBuffer:bp offset:0 atIndex:2]; [encoder setBuffer:norms offset:0 atIndex:3];
		SET_SCALAR(count, uint, 4); SET_SCALAR(learning_rate, float, 5);
		rc = run_threads(pipeline, command_buffer, encoder, count);
	}

	if (rc == 0) finish_state(out, bo, moment, bm, variance, bv, count);
	free(m); free(v); free(p); free(g); return rc;
}

int metal_optimizer_adagrad(double *out, double *accumulator, const double *params, const double *grads, int count, double learning_rate, double eps, double weight_decay) {
	float *a = to_float(accumulator, count), *p = to_float(params, count), *g = to_float(grads, count);
	START_ENCODER("adagrad");
	id<MTLBuffer> bo = rw(NULL, count), ba = rw(a, count), bp = ro(p, count), bg = ro(g, count);
	[encoder setBuffer:bo offset:0 atIndex:0]; [encoder setBuffer:ba offset:0 atIndex:1]; [encoder setBuffer:bp offset:0 atIndex:2]; [encoder setBuffer:bg offset:0 atIndex:3];
	SET_SCALAR(count, uint, 4); SET_SCALAR(learning_rate, float, 5); SET_SCALAR(eps, float, 6); SET_SCALAR(weight_decay, float, 7);
	int rc = run_threads(pipeline, command_buffer, encoder, count);
	if (rc == 0) finish_state(out, bo, accumulator, ba, NULL, nil, count);
	free(a); free(p); free(g); return rc;
}

int metal_optimizer_adadelta(double *out, double *grad_average, double *delta_average, const double *params, const double *grads, int count, double rho, double eps, double weight_decay) {
	float *ga = to_float(grad_average, count), *da = to_float(delta_average, count), *p = to_float(params, count), *g = to_float(grads, count);
	START_ENCODER("adadelta");
	id<MTLBuffer> bo = rw(NULL, count), bga = rw(ga, count), bda = rw(da, count), bp = ro(p, count), bg = ro(g, count);
	[encoder setBuffer:bo offset:0 atIndex:0]; [encoder setBuffer:bga offset:0 atIndex:1]; [encoder setBuffer:bda offset:0 atIndex:2]; [encoder setBuffer:bp offset:0 atIndex:3]; [encoder setBuffer:bg offset:0 atIndex:4];
	SET_SCALAR(count, uint, 5); SET_SCALAR(rho, float, 6); SET_SCALAR(eps, float, 7); SET_SCALAR(weight_decay, float, 8);
	int rc = run_threads(pipeline, command_buffer, encoder, count);
	if (rc == 0) { to_double(out, [bo contents], count); to_double(grad_average, [bga contents], count); to_double(delta_average, [bda contents], count); }
	free(ga); free(da); free(p); free(g); return rc;
}

int metal_optimizer_lbfgs(double *out, double *s_history, double *y_history, double *rho_history, int *head, int *history_count, const double *params, const double *grads, const double *previous_params, const double *previous_grads, int has_previous, int count, int history_size, double learning_rate, int line_search, double c1) {
	int history_elements = count * history_size;
	float *sh = to_float(s_history, history_elements), *yh = to_float(y_history, history_elements), *rh = to_float(rho_history, history_size), *p = to_float(params, count), *g = to_float(grads, count), *pp = to_float(previous_params, count), *pg = to_float(previous_grads, count);
	START_ENCODER("lbfgs");
	id<MTLBuffer> bo = rw(NULL, count), bsh = rw(sh, history_elements), byh = rw(yh, history_elements), brh = rw(rh, history_size), bd = rw(NULL, count), ba = rw(NULL, history_size), bp = ro(p, count), bg = ro(g, count), bpp = ro(pp, count), bpg = ro(pg, count);
	uint h = (uint)*head, hc = (uint)*history_count;
	id<MTLBuffer> bh = rw_uint(&h, 1), bhc = rw_uint(&hc, 1);
	[encoder setBuffer:bo offset:0 atIndex:0]; [encoder setBuffer:bsh offset:0 atIndex:1]; [encoder setBuffer:byh offset:0 atIndex:2]; [encoder setBuffer:brh offset:0 atIndex:3]; [encoder setBuffer:bd offset:0 atIndex:4]; [encoder setBuffer:ba offset:0 atIndex:5]; [encoder setBuffer:bh offset:0 atIndex:6]; [encoder setBuffer:bhc offset:0 atIndex:7]; [encoder setBuffer:bp offset:0 atIndex:8]; [encoder setBuffer:bg offset:0 atIndex:9]; [encoder setBuffer:bpp offset:0 atIndex:10]; [encoder setBuffer:bpg offset:0 atIndex:11];
	SET_SCALAR(has_previous, uint, 12); SET_SCALAR(count, uint, 13); SET_SCALAR(history_size, uint, 14); SET_SCALAR(learning_rate, float, 15); SET_SCALAR(line_search, uint, 16); SET_SCALAR(c1, float, 17);
	int rc = run_threads(pipeline, command_buffer, encoder, 1);
	if (rc == 0) { to_double(out, [bo contents], count); to_double(s_history, [bsh contents], history_elements); to_double(y_history, [byh contents], history_elements); to_double(rho_history, [brh contents], history_size); *head = (int)(*((uint *)[bh contents])); *history_count = (int)(*((uint *)[bhc contents])); }
	free(sh); free(yh); free(rh); free(p); free(g); free(pp); free(pg); return rc;
}
