#include <metal_stdlib>
using namespace metal;

kernel void adam(
	device float *out [[buffer(0)]],
	device float *moment [[buffer(1)]],
	device float *variance [[buffer(2)]],
	device const float *params [[buffer(3)]],
	device const float *grads [[buffer(4)]],
	constant uint &count [[buffer(5)]],
	constant float &beta1 [[buffer(6)]],
	constant float &beta2 [[buffer(7)]],
	constant float &learning_rate [[buffer(8)]],
	constant float &eps [[buffer(9)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float grad = grads[index];
	float next_moment = beta1 * moment[index] + (1.0f - beta1) * grad;
	float next_variance = beta2 * variance[index] + (1.0f - beta2) * grad * grad;

	moment[index] = next_moment;
	variance[index] = next_variance;
	out[index] = params[index] - learning_rate * next_moment / (sqrt(next_variance) + eps);
}

kernel void adamw(
	device float *out [[buffer(0)]],
	device float *moment [[buffer(1)]],
	device float *variance [[buffer(2)]],
	device const float *params [[buffer(3)]],
	device const float *grads [[buffer(4)]],
	constant uint &count [[buffer(5)]],
	constant float &beta1 [[buffer(6)]],
	constant float &beta2 [[buffer(7)]],
	constant float &learning_rate [[buffer(8)]],
	constant float &eps [[buffer(9)]],
	constant float &weight_decay_step [[buffer(10)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float grad = grads[index];
	float next_moment = beta1 * moment[index] + (1.0f - beta1) * grad;
	float next_variance = beta2 * variance[index] + (1.0f - beta2) * grad * grad;

	moment[index] = next_moment;
	variance[index] = next_variance;
	out[index] = params[index] - weight_decay_step * params[index] -
		learning_rate * next_moment / (sqrt(next_variance) + eps);
}

kernel void adamax(
	device float *out [[buffer(0)]],
	device float *moment [[buffer(1)]],
	device float *infinity_norm [[buffer(2)]],
	device const float *params [[buffer(3)]],
	device const float *grads [[buffer(4)]],
	constant uint &count [[buffer(5)]],
	constant float &beta1 [[buffer(6)]],
	constant float &beta2 [[buffer(7)]],
	constant float &learning_rate [[buffer(8)]],
	constant float &eps [[buffer(9)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float grad = grads[index];
	float next_moment = beta1 * moment[index] + (1.0f - beta1) * grad;
	float next_norm = max(beta2 * infinity_norm[index], abs(grad));

	moment[index] = next_moment;
	infinity_norm[index] = next_norm;
	out[index] = params[index] - learning_rate * next_moment / (next_norm + eps);
}

kernel void sgd(
	device float *out [[buffer(0)]],
	device float *velocity [[buffer(1)]],
	device const float *params [[buffer(2)]],
	device const float *grads [[buffer(3)]],
	constant uint &count [[buffer(4)]],
	constant float &learning_rate [[buffer(5)]],
	constant float &weight_decay [[buffer(6)]],
	constant float &momentum [[buffer(7)]],
	constant uint &nesterov [[buffer(8)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float grad = grads[index] + weight_decay * params[index];

	if (momentum == 0.0f) {
		out[index] = params[index] - learning_rate * grad;
		return;
	}

	float next_velocity = momentum * velocity[index] + grad;
	velocity[index] = next_velocity;
	float update = nesterov != 0 ? grad + momentum * next_velocity : next_velocity;
	out[index] = params[index] - learning_rate * update;
}

kernel void lion(
	device float *out [[buffer(0)]],
	device float *moment [[buffer(1)]],
	device const float *params [[buffer(2)]],
	device const float *grads [[buffer(3)]],
	constant uint &count [[buffer(4)]],
	constant float &learning_rate [[buffer(5)]],
	constant float &beta1 [[buffer(6)]],
	constant float &beta2 [[buffer(7)]],
	constant float &weight_decay [[buffer(8)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float blended = beta1 * moment[index] + (1.0f - beta1) * grads[index];
	float sign_value = blended > 0.0f ? 1.0f : (blended < 0.0f ? -1.0f : 0.0f);

	out[index] = params[index] - learning_rate * (sign_value + weight_decay * params[index]);
	moment[index] = beta2 * moment[index] + (1.0f - beta2) * grads[index];
}

kernel void rmsprop(
	device float *out [[buffer(0)]],
	device float *square_average [[buffer(1)]],
	device float *momentum_buffer [[buffer(2)]],
	device float *grad_average [[buffer(3)]],
	device const float *params [[buffer(4)]],
	device const float *grads [[buffer(5)]],
	constant uint &count [[buffer(6)]],
	constant float &learning_rate [[buffer(7)]],
	constant float &alpha [[buffer(8)]],
	constant float &eps [[buffer(9)]],
	constant float &momentum [[buffer(10)]],
	constant float &weight_decay [[buffer(11)]],
	constant uint &centered [[buffer(12)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float grad = grads[index] + weight_decay * params[index];
	float next_square = alpha * square_average[index] + (1.0f - alpha) * grad * grad;
	square_average[index] = next_square;

	float average = next_square;

	if (centered != 0) {
		float next_grad_average = alpha * grad_average[index] + (1.0f - alpha) * grad;
		grad_average[index] = next_grad_average;
		average -= next_grad_average * next_grad_average;
	}

	float update = grad / (sqrt(average) + eps);

	if (momentum != 0.0f) {
		update = momentum * momentum_buffer[index] + update;
		momentum_buffer[index] = update;
	}

	out[index] = params[index] - learning_rate * update;
}

kernel void hebbian(
	device float *out [[buffer(0)]],
	device atomic_float *norm_square [[buffer(1)]],
	device const float *params [[buffer(2)]],
	device const float *grads [[buffer(3)]],
	constant uint &count [[buffer(4)]],
	constant float &learning_rate [[buffer(5)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float value = params[index] + learning_rate * grads[index];
	out[index] = value;
	atomic_fetch_add_explicit(norm_square, value * value, memory_order_relaxed);
}

kernel void scale(
	device float *out [[buffer(0)]],
	constant uint &count [[buffer(1)]],
	constant float &factor [[buffer(2)]],
	uint index [[thread_position_in_grid]])
{
	if (index < count) out[index] *= factor;
}

kernel void scale_by_norm_limit(
	device float *out [[buffer(0)]],
	device const float *norm_square [[buffer(1)]],
	constant uint &count [[buffer(2)]],
	constant float &max_norm [[buffer(3)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count || max_norm <= 0.0f) return;

	float norm = sqrt(norm_square[0]);

	if (norm <= max_norm || norm <= 0.0f) return;

	out[index] *= max_norm / norm;
}

kernel void lars_norms(
	device const float *params [[buffer(0)]],
	device const float *grads [[buffer(1)]],
	device float2 *partials [[buffer(2)]],
	constant uint &count [[buffer(3)]],
	uint grid_index [[thread_position_in_grid]],
	uint local_index [[thread_index_in_threadgroup]],
	uint group_position [[threadgroup_position_in_grid]])
{
	threadgroup float param_squares[256];
	threadgroup float grad_squares[256];

	float param_square = 0.0f;
	float grad_square = 0.0f;

	if (grid_index < count) {
		float param = params[grid_index];
		float grad = grads[grid_index];
		param_square = param * param;
		grad_square = grad * grad;
	}

	param_squares[local_index] = param_square;
	grad_squares[local_index] = grad_square;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (local_index < stride) {
			param_squares[local_index] += param_squares[local_index + stride];
			grad_squares[local_index] += grad_squares[local_index + stride];
		}

		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	if (local_index == 0) {
		partials[group_position] = float2(param_squares[0], grad_squares[0]);
	}
}

kernel void reduce_pair_sums(
	device const float2 *input [[buffer(0)]],
	device float2 *output [[buffer(1)]],
	constant uint &count [[buffer(2)]],
	uint grid_index [[thread_position_in_grid]],
	uint local_index [[thread_index_in_threadgroup]],
	uint group_position [[threadgroup_position_in_grid]])
{
	threadgroup float left_sums[256];
	threadgroup float right_sums[256];

	float2 value = float2(0.0f, 0.0f);

	if (grid_index < count) {
		value = input[grid_index];
	}

	left_sums[local_index] = value.x;
	right_sums[local_index] = value.y;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (local_index < stride) {
			left_sums[local_index] += left_sums[local_index + stride];
			right_sums[local_index] += right_sums[local_index + stride];
		}

		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	if (local_index == 0) {
		output[group_position] = float2(left_sums[0], right_sums[0]);
	}
}

kernel void lars_apply(
	device float *out [[buffer(0)]],
	device float *velocity [[buffer(1)]],
	device const float *params [[buffer(2)]],
	device const float *grads [[buffer(3)]],
	device const float2 *norms [[buffer(4)]],
	constant uint &count [[buffer(5)]],
	constant float &learning_rate [[buffer(6)]],
	constant float &eta [[buffer(7)]],
	constant float &momentum [[buffer(8)]],
	constant float &weight_decay [[buffer(9)]],
	constant float &eps [[buffer(10)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float param_norm = sqrt(norms[0].x);
	float grad_norm = sqrt(norms[0].y);
	float local_learning_rate = learning_rate;

	if (param_norm > 0.0f && grad_norm > 0.0f) {
		local_learning_rate = eta * param_norm /
			(grad_norm + weight_decay * param_norm + eps);
	}

	float grad = grads[index] + weight_decay * params[index];
	float next_velocity = momentum * velocity[index] + local_learning_rate * grad;
	velocity[index] = next_velocity;
	out[index] = params[index] - next_velocity;
}

kernel void lamb_prepare(
	device float *update [[buffer(0)]],
	device float *moment [[buffer(1)]],
	device float *variance [[buffer(2)]],
	device const float *params [[buffer(3)]],
	device const float *grads [[buffer(4)]],
	device float2 *partials [[buffer(5)]],
	constant uint &count [[buffer(6)]],
	constant float &beta1 [[buffer(7)]],
	constant float &beta2 [[buffer(8)]],
	constant float &eps [[buffer(9)]],
	constant float &weight_decay [[buffer(10)]],
	constant float &bias_correction1_inv [[buffer(11)]],
	constant float &bias_correction2_inv [[buffer(12)]],
	uint grid_index [[thread_position_in_grid]],
	uint local_index [[thread_index_in_threadgroup]],
	uint group_position [[threadgroup_position_in_grid]])
{
	threadgroup float param_squares[256];
	threadgroup float update_squares[256];

	float param_square = 0.0f;
	float update_square = 0.0f;

	if (grid_index < count) {
		float grad = grads[grid_index];
		float param = params[grid_index];
		float next_moment = beta1 * moment[grid_index] + (1.0f - beta1) * grad;
		float next_variance = beta2 * variance[grid_index] + (1.0f - beta2) * grad * grad;
		float next_update = next_moment * bias_correction1_inv /
			(sqrt(next_variance * bias_correction2_inv) + eps) + weight_decay * param;

		moment[grid_index] = next_moment;
		variance[grid_index] = next_variance;
		update[grid_index] = next_update;
		param_square = param * param;
		update_square = next_update * next_update;
	}

	param_squares[local_index] = param_square;
	update_squares[local_index] = update_square;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (local_index < stride) {
			param_squares[local_index] += param_squares[local_index + stride];
			update_squares[local_index] += update_squares[local_index + stride];
		}

		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	if (local_index == 0) {
		partials[group_position] = float2(param_squares[0], update_squares[0]);
	}
}

kernel void lamb_apply(
	device float *out [[buffer(0)]],
	device const float *update [[buffer(1)]],
	device const float *params [[buffer(2)]],
	device const float2 *norms [[buffer(3)]],
	constant uint &count [[buffer(4)]],
	constant float &learning_rate [[buffer(5)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float ratio = learning_rate;

	if (norms[0].x > 0.0f && norms[0].y > 0.0f) {
		ratio = learning_rate * sqrt(norms[0].x) / sqrt(norms[0].y);
	}

	out[index] = params[index] - ratio * update[index];
}

kernel void adagrad(
	device float *out [[buffer(0)]],
	device float *accumulator [[buffer(1)]],
	device const float *params [[buffer(2)]],
	device const float *grads [[buffer(3)]],
	constant uint &count [[buffer(4)]],
	constant float &learning_rate [[buffer(5)]],
	constant float &eps [[buffer(6)]],
	constant float &weight_decay [[buffer(7)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float grad = grads[index] + weight_decay * params[index];
	accumulator[index] += grad * grad;
	out[index] = params[index] - learning_rate * grad / (sqrt(accumulator[index]) + eps);
}

kernel void adadelta(
	device float *out [[buffer(0)]],
	device float *grad_average [[buffer(1)]],
	device float *delta_average [[buffer(2)]],
	device const float *params [[buffer(3)]],
	device const float *grads [[buffer(4)]],
	constant uint &count [[buffer(5)]],
	constant float &rho [[buffer(6)]],
	constant float &eps [[buffer(7)]],
	constant float &weight_decay [[buffer(8)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float grad = grads[index] + weight_decay * params[index];
	grad_average[index] = rho * grad_average[index] + (1.0f - rho) * grad * grad;
	float update = sqrt(delta_average[index] + eps) /
		sqrt(grad_average[index] + eps) * grad;
	out[index] = params[index] - update;
	delta_average[index] = rho * delta_average[index] + (1.0f - rho) * update * update;
}

kernel void lbfgs_history_delta(
	device float *state_history [[buffer(0)]],
	device float *grad_history [[buffer(1)]],
	device float2 *partials [[buffer(2)]],
	device const float *params [[buffer(3)]],
	device const float *grads [[buffer(4)]],
	device const float *previous_params [[buffer(5)]],
	device const float *previous_grads [[buffer(6)]],
	device const float *head [[buffer(7)]],
	constant uint &count [[buffer(8)]],
	constant uint &history_size [[buffer(9)]],
	uint grid_index [[thread_position_in_grid]],
	uint local_index [[thread_index_in_threadgroup]],
	uint group_position [[threadgroup_position_in_grid]])
{
	threadgroup float curvature_sums[256];

	uint slot = uint(head[0]) % history_size;
	float curvature = 0.0f;

	if (grid_index < count) {
		float state_delta = params[grid_index] - previous_params[grid_index];
		float grad_delta = grads[grid_index] - previous_grads[grid_index];
		state_history[slot * count + grid_index] = state_delta;
		grad_history[slot * count + grid_index] = grad_delta;
		curvature = grad_delta * state_delta;
	}

	curvature_sums[local_index] = curvature;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (local_index < stride) {
			curvature_sums[local_index] += curvature_sums[local_index + stride];
		}

		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	if (local_index == 0) {
		partials[group_position] = float2(curvature_sums[0], 0.0f);
	}
}

kernel void lbfgs_accept_history(
	device float *rho_history [[buffer(0)]],
	device float *head [[buffer(1)]],
	device float *history_count [[buffer(2)]],
	device const float2 *curvature [[buffer(3)]],
	constant uint &history_size [[buffer(4)]],
	uint index [[thread_position_in_grid]])
{
	if (index != 0) return;

	float value = curvature[0].x;

	if (value <= 1e-10f) return;

	uint current_head = uint(head[0]);
	uint current_count = uint(history_count[0]);
	uint slot = current_head % history_size;
	rho_history[slot] = 1.0f / value;
	head[0] = float(current_head + 1);

	if (current_count < history_size) {
		history_count[0] = float(current_count + 1);
	}
}

kernel void lbfgs_direction_init(
	device float *direction [[buffer(0)]],
	device const float *grads [[buffer(1)]],
	constant uint &count [[buffer(2)]],
	uint index [[thread_position_in_grid]])
{
	if (index < count) direction[index] = grads[index];
}

kernel void lbfgs_dot(
	device const float *left [[buffer(0)]],
	device const float *right [[buffer(1)]],
	device float2 *partials [[buffer(2)]],
	constant uint &count [[buffer(3)]],
	uint grid_index [[thread_position_in_grid]],
	uint local_index [[thread_index_in_threadgroup]],
	uint group_position [[threadgroup_position_in_grid]])
{
	threadgroup float dot_sums[256];

	dot_sums[local_index] = grid_index < count ? left[grid_index] * right[grid_index] : 0.0f;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (local_index < stride) {
			dot_sums[local_index] += dot_sums[local_index + stride];
		}

		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	if (local_index == 0) {
		partials[group_position] = float2(dot_sums[0], 0.0f);
	}
}

kernel void lbfgs_gamma(
	device const float *y_slot [[buffer(0)]],
	device const float *state_slot [[buffer(1)]],
	device float2 *partials [[buffer(2)]],
	constant uint &count [[buffer(3)]],
	uint grid_index [[thread_position_in_grid]],
	uint local_index [[thread_index_in_threadgroup]],
	uint group_position [[threadgroup_position_in_grid]])
{
	threadgroup float yy_sums[256];
	threadgroup float ys_sums[256];

	float yy = 0.0f;
	float ys = 0.0f;

	if (grid_index < count) {
		float y_value = y_slot[grid_index];
		yy = y_value * y_value;
		ys = y_value * state_slot[grid_index];
	}

	yy_sums[local_index] = yy;
	ys_sums[local_index] = ys;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (local_index < stride) {
			yy_sums[local_index] += yy_sums[local_index + stride];
			ys_sums[local_index] += ys_sums[local_index + stride];
		}

		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	if (local_index == 0) {
		partials[group_position] = float2(yy_sums[0], ys_sums[0]);
	}
}

kernel void lbfgs_store_alpha(
	device float *alphas [[buffer(0)]],
	device const float *rho_history [[buffer(1)]],
	device const float2 *dot [[buffer(2)]],
	constant uint &history_index [[buffer(3)]],
	constant uint &slot [[buffer(4)]],
	uint index [[thread_position_in_grid]])
{
	if (index == 0) alphas[history_index] = rho_history[slot] * dot[0].x;
}

kernel void lbfgs_reverse_apply(
	device float *direction [[buffer(0)]],
	device const float *y_slot [[buffer(1)]],
	device const float *alphas [[buffer(2)]],
	constant uint &history_index [[buffer(3)]],
	constant uint &count [[buffer(4)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	direction[index] -= alphas[history_index] * y_slot[index];
}

kernel void lbfgs_gamma_apply(
	device float *direction [[buffer(0)]],
	device const float2 *gamma_pair [[buffer(1)]],
	constant uint &count [[buffer(2)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float gamma = gamma_pair[0].x == 0.0f ? 1.0f : gamma_pair[0].y / gamma_pair[0].x;
	direction[index] *= gamma;
}

kernel void lbfgs_forward_apply(
	device float *direction [[buffer(0)]],
	device const float *state_slot [[buffer(1)]],
	device const float *rho_history [[buffer(2)]],
	device const float *alphas [[buffer(3)]],
	device const float2 *dot [[buffer(4)]],
	constant uint &history_index [[buffer(5)]],
	constant uint &slot [[buffer(6)]],
	constant uint &count [[buffer(7)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float beta = rho_history[slot] * dot[0].x;
	direction[index] += (alphas[history_index] - beta) * state_slot[index];
}

kernel void lbfgs_line_search(
	device const float *grads [[buffer(0)]],
	device const float *direction [[buffer(1)]],
	device float2 *partials [[buffer(2)]],
	constant uint &count [[buffer(3)]],
	uint grid_index [[thread_position_in_grid]],
	uint local_index [[thread_index_in_threadgroup]],
	uint group_position [[threadgroup_position_in_grid]])
{
	threadgroup float f_sums[256];
	threadgroup float slope_sums[256];

	float f_value = 0.0f;
	float slope_value = 0.0f;

	if (grid_index < count) {
		float grad = grads[grid_index];
		f_value = grad * grad;
		slope_value = grad * direction[grid_index];
	}

	f_sums[local_index] = f_value;
	slope_sums[local_index] = slope_value;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (local_index < stride) {
			f_sums[local_index] += f_sums[local_index + stride];
			slope_sums[local_index] += slope_sums[local_index + stride];
		}

		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	if (local_index == 0) {
		partials[group_position] = float2(f_sums[0], slope_sums[0]);
	}
}

kernel void lbfgs_finalize(
	device float *out [[buffer(0)]],
	device const float *params [[buffer(1)]],
	device const float *direction [[buffer(2)]],
	device const float2 *line_metrics [[buffer(3)]],
	constant uint &count [[buffer(4)]],
	constant float &learning_rate [[buffer(5)]],
	constant uint &line_search [[buffer(6)]],
	constant float &c1 [[buffer(7)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float effective_learning_rate = learning_rate;

	if (line_search != 0) {
		float f0 = line_metrics[0].x;
		float slope = -line_metrics[0].y;
		float c1_value = c1 == 0.0f ? 1e-4f : c1;

		for (uint search_index = 0; search_index < 50; search_index++) {
			float decrease = f0 - c1_value * effective_learning_rate * slope;

			if (decrease > 0.0f) break;

			effective_learning_rate *= 0.5f;

			if (effective_learning_rate < 1e-10f) break;
		}

		if (effective_learning_rate < 1e-10f) {
			effective_learning_rate = 1e-10f;
		}
	}

	out[index] = params[index] - effective_learning_rate * direction[index];
}
