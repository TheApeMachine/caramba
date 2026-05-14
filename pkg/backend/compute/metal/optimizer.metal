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

kernel void lars(
	device float *out [[buffer(0)]],
	device float *velocity [[buffer(1)]],
	device const float *params [[buffer(2)]],
	device const float *grads [[buffer(3)]],
	constant uint &count [[buffer(4)]],
	constant float &learning_rate [[buffer(5)]],
	constant float &eta [[buffer(6)]],
	constant float &momentum [[buffer(7)]],
	constant float &weight_decay [[buffer(8)]],
	constant float &eps [[buffer(9)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float param_norm_square = 0.0f;
	float grad_norm_square = 0.0f;

	for (uint norm_index = 0; norm_index < count; norm_index++) {
		param_norm_square += params[norm_index] * params[norm_index];
		grad_norm_square += grads[norm_index] * grads[norm_index];
	}

	float param_norm = sqrt(param_norm_square);
	float grad_norm = sqrt(grad_norm_square);
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

kernel void lamb(
	device float *out [[buffer(0)]],
	device float *moment [[buffer(1)]],
	device float *variance [[buffer(2)]],
	device const float *params [[buffer(3)]],
	device const float *grads [[buffer(4)]],
	constant uint &count [[buffer(5)]],
	constant float &learning_rate [[buffer(6)]],
	constant float &beta1 [[buffer(7)]],
	constant float &beta2 [[buffer(8)]],
	constant float &eps [[buffer(9)]],
	constant float &weight_decay [[buffer(10)]],
	constant float &bias_correction1_inv [[buffer(11)]],
	constant float &bias_correction2_inv [[buffer(12)]],
	uint index [[thread_position_in_grid]])
{
	if (index >= count) return;

	float param_norm_square = 0.0f;
	float update_norm_square = 0.0f;

	for (uint norm_index = 0; norm_index < count; norm_index++) {
		float next_moment = beta1 * moment[norm_index] +
			(1.0f - beta1) * grads[norm_index];
		float next_variance = beta2 * variance[norm_index] +
			(1.0f - beta2) * grads[norm_index] * grads[norm_index];
		float update = next_moment * bias_correction1_inv /
			(sqrt(next_variance * bias_correction2_inv) + eps) +
			weight_decay * params[norm_index];
		param_norm_square += params[norm_index] * params[norm_index];
		update_norm_square += update * update;
	}

	float grad = grads[index];
	float next_moment = beta1 * moment[index] + (1.0f - beta1) * grad;
	float next_variance = beta2 * variance[index] + (1.0f - beta2) * grad * grad;
	float update = next_moment * bias_correction1_inv /
		(sqrt(next_variance * bias_correction2_inv) + eps) +
		weight_decay * params[index];
	float ratio = learning_rate;

	if (param_norm_square > 0.0f && update_norm_square > 0.0f) {
		ratio = learning_rate * sqrt(param_norm_square) / sqrt(update_norm_square);
	}

	moment[index] = next_moment;
	variance[index] = next_variance;
	out[index] = params[index] - ratio * update;
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

kernel void lbfgs(
	device float *out [[buffer(0)]],
	device float *state_history [[buffer(1)]],
	device float *grad_history [[buffer(2)]],
	device float *rho_history [[buffer(3)]],
	device uint *head [[buffer(4)]],
	device uint *history_count [[buffer(5)]],
	device const float *params [[buffer(6)]],
	device const float *grads [[buffer(7)]],
	device const float *previous_params [[buffer(8)]],
	device const float *previous_grads [[buffer(9)]],
	constant uint &has_previous [[buffer(10)]],
	constant uint &count [[buffer(11)]],
	constant uint &history_size [[buffer(12)]],
	constant float &learning_rate [[buffer(13)]],
	uint index [[thread_position_in_grid]])
{
	if (index != 0) return;

	if (has_previous != 0 && history_size > 0) {
		uint slot = head[0] % history_size;
		float curvature = 0.0f;

		for (uint value_index = 0; value_index < count; value_index++) {
			float state_delta = params[value_index] - previous_params[value_index];
			float grad_delta = grads[value_index] - previous_grads[value_index];
			state_history[slot * count + value_index] = state_delta;
			grad_history[slot * count + value_index] = grad_delta;
			curvature += grad_delta * state_delta;
		}

		if (curvature > 1e-10f) {
			rho_history[slot] = 1.0f / curvature;
			head[0]++;
			if (history_count[0] < history_size) history_count[0]++;
		}
	}

	for (uint value_index = 0; value_index < count; value_index++) {
		out[value_index] = params[value_index] - learning_rate * grads[value_index];
	}
}
