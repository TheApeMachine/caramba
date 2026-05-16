#include "optimizer.h"

#include <cmath>
#include <cstring>
#include <vector>

/*
These entry points are intentionally shaped as fused optimizer kernels for the
XLA backend ABI. They are kept separate from CPU optimizer code and are compiled
only inside the XLA backend build.
*/

extern "C" {

int xla_optimizer_adam(
	double *out, double *moment, double *variance,
	const double *params, const double *grads, int count,
	double beta1, double beta2, double learning_rate, double eps
) {
	for (int index = 0; index < count; index++) {
		double grad = grads[index];
		double next_moment = beta1 * moment[index] + (1.0 - beta1) * grad;
		double next_variance = beta2 * variance[index] + (1.0 - beta2) * grad * grad;
		moment[index] = next_moment;
		variance[index] = next_variance;
		out[index] = params[index] - learning_rate * next_moment / (std::sqrt(next_variance) + eps);
	}

	return 0;
}

int xla_optimizer_adamw(
	double *out, double *moment, double *variance,
	const double *params, const double *grads, int count,
	double beta1, double beta2, double learning_rate, double eps,
	double weight_decay_step
) {
	for (int index = 0; index < count; index++) {
		double grad = grads[index];
		double next_moment = beta1 * moment[index] + (1.0 - beta1) * grad;
		double next_variance = beta2 * variance[index] + (1.0 - beta2) * grad * grad;
		moment[index] = next_moment;
		variance[index] = next_variance;
		out[index] = params[index] - weight_decay_step * params[index] -
			learning_rate * next_moment / (std::sqrt(next_variance) + eps);
	}

	return 0;
}

int xla_optimizer_adamax(
	double *out, double *moment, double *infinity_norm,
	const double *params, const double *grads, int count,
	double beta1, double beta2, double learning_rate, double eps
) {
	for (int index = 0; index < count; index++) {
		double grad = grads[index];
		double next_moment = beta1 * moment[index] + (1.0 - beta1) * grad;
		double next_norm = std::fmax(beta2 * infinity_norm[index], std::fabs(grad));
		moment[index] = next_moment;
		infinity_norm[index] = next_norm;
		out[index] = params[index] - learning_rate * next_moment / (next_norm + eps);
	}

	return 0;
}

int xla_optimizer_sgd(
	double *out, double *velocity,
	const double *params, const double *grads, int count,
	double learning_rate, double weight_decay, double momentum, int nesterov
) {
	for (int index = 0; index < count; index++) {
		double grad = grads[index] + weight_decay * params[index];
		if (momentum == 0.0) {
			out[index] = params[index] - learning_rate * grad;
			continue;
		}
		double next_velocity = momentum * velocity[index] + grad;
		velocity[index] = next_velocity;
		double update = nesterov != 0 ? grad + momentum * next_velocity : next_velocity;
		out[index] = params[index] - learning_rate * update;
	}

	return 0;
}

int xla_optimizer_lion(
	double *out, double *moment,
	const double *params, const double *grads, int count,
	double learning_rate, double beta1, double beta2, double weight_decay
) {
	for (int index = 0; index < count; index++) {
		double blended = beta1 * moment[index] + (1.0 - beta1) * grads[index];
		double sign = blended > 0.0 ? 1.0 : (blended < 0.0 ? -1.0 : 0.0);
		out[index] = params[index] - learning_rate * (sign + weight_decay * params[index]);
		moment[index] = beta2 * moment[index] + (1.0 - beta2) * grads[index];
	}

	return 0;
}

int xla_optimizer_rmsprop(
	double *out, double *square_average, double *momentum_buffer,
	double *grad_average, const double *params, const double *grads,
	int count, double learning_rate, double alpha, double eps,
	double momentum, double weight_decay, int centered
) {
	for (int index = 0; index < count; index++) {
		double grad = grads[index] + weight_decay * params[index];
		double next_square = alpha * square_average[index] + (1.0 - alpha) * grad * grad;
		square_average[index] = next_square;
		double average = next_square;
		if (centered != 0) {
			double next_grad_average = alpha * grad_average[index] + (1.0 - alpha) * grad;
			grad_average[index] = next_grad_average;
			average -= next_grad_average * next_grad_average;
		}
		double update = grad / (std::sqrt(average) + eps);
		if (momentum != 0.0) {
			update = momentum * momentum_buffer[index] + update;
			momentum_buffer[index] = update;
		}
		out[index] = params[index] - learning_rate * update;
	}

	return 0;
}

int xla_optimizer_hebbian(
	double *out, const double *params, const double *grads, int count,
	double learning_rate, double max_norm
) {
	double norm_square = 0.0;
	for (int index = 0; index < count; index++) {
		out[index] = params[index] + learning_rate * grads[index];
		norm_square += out[index] * out[index];
	}
	double norm = std::sqrt(norm_square);
	if (max_norm > 0.0 && norm > max_norm) {
		double scale = max_norm / norm;
		for (int index = 0; index < count; index++) out[index] *= scale;
	}

	return 0;
}

int xla_optimizer_lars(
	double *out, double *velocity,
	const double *params, const double *grads, int count,
	double learning_rate, double eta, double momentum,
	double weight_decay, double eps
) {
	double param_norm_square = 0.0;
	double grad_norm_square = 0.0;
	for (int index = 0; index < count; index++) {
		param_norm_square += params[index] * params[index];
		grad_norm_square += grads[index] * grads[index];
	}
	double param_norm = std::sqrt(param_norm_square);
	double grad_norm = std::sqrt(grad_norm_square);
	double local_learning_rate = learning_rate;
	if (param_norm > 0.0 && grad_norm > 0.0) {
		local_learning_rate = eta * param_norm / (grad_norm + weight_decay * param_norm + eps);
	}
	for (int index = 0; index < count; index++) {
		double grad = grads[index] + weight_decay * params[index];
		velocity[index] = momentum * velocity[index] + local_learning_rate * grad;
		out[index] = params[index] - velocity[index];
	}

	return 0;
}

int xla_optimizer_lamb(
	double *out, double *moment, double *variance,
	const double *params, const double *grads, int count,
	double learning_rate, double beta1, double beta2, double eps,
	double weight_decay, double bias_correction1_inv,
	double bias_correction2_inv
) {
	double param_norm_square = 0.0;
	double update_norm_square = 0.0;
	for (int index = 0; index < count; index++) {
		double next_moment = beta1 * moment[index] + (1.0 - beta1) * grads[index];
		double next_variance = beta2 * variance[index] + (1.0 - beta2) * grads[index] * grads[index];
		double update = next_moment * bias_correction1_inv /
			(std::sqrt(next_variance * bias_correction2_inv) + eps) + weight_decay * params[index];
		out[index] = update;
		moment[index] = next_moment;
		variance[index] = next_variance;
		param_norm_square += params[index] * params[index];
		update_norm_square += update * update;
	}
	double ratio = learning_rate;
	if (param_norm_square > 0.0 && update_norm_square > 0.0) {
		ratio = learning_rate * std::sqrt(param_norm_square) / std::sqrt(update_norm_square);
	}
	for (int index = 0; index < count; index++) out[index] = params[index] - ratio * out[index];

	return 0;
}

int xla_optimizer_adagrad(
	double *out, double *accumulator,
	const double *params, const double *grads, int count,
	double learning_rate, double eps, double weight_decay
) {
	for (int index = 0; index < count; index++) {
		double grad = grads[index] + weight_decay * params[index];
		accumulator[index] += grad * grad;
		out[index] = params[index] - learning_rate * grad / (std::sqrt(accumulator[index]) + eps);
	}

	return 0;
}

int xla_optimizer_adadelta(
	double *out, double *grad_average, double *delta_average,
	const double *params, const double *grads, int count,
	double rho, double eps, double weight_decay
) {
	for (int index = 0; index < count; index++) {
		double grad = grads[index] + weight_decay * params[index];
		grad_average[index] = rho * grad_average[index] + (1.0 - rho) * grad * grad;
		double update = std::sqrt(delta_average[index] + eps) /
			std::sqrt(grad_average[index] + eps) * grad;
		out[index] = params[index] - update;
		delta_average[index] = rho * delta_average[index] + (1.0 - rho) * update * update;
	}

	return 0;
}

int xla_optimizer_lbfgs(
	double *out, double *s_history, double *y_history, double *rho_history,
	int *head, int *history_count, const double *params,
	const double *grads, const double *previous_params,
	const double *previous_grads, int has_previous, int count,
	int history_size, double learning_rate, int line_search, double c1
) {
	if (has_previous != 0 && history_size > 0) {
		int slot = (*head) % history_size;
		double curvature = 0.0;

		for (int index = 0; index < count; index++) {
			double state_delta = params[index] - previous_params[index];
			double grad_delta = grads[index] - previous_grads[index];
			s_history[slot * count + index] = state_delta;
			y_history[slot * count + index] = grad_delta;
			curvature += grad_delta * state_delta;
		}

		if (curvature > 1e-10) {
			rho_history[slot] = 1.0 / curvature;
			*head += 1;
			if (*history_count < history_size) *history_count += 1;
		}
	}

	std::vector<double> direction((size_t)count);
	std::vector<double> alphas((size_t)*history_count);

	for (int index = 0; index < count; index++) {
		direction[(size_t)index] = grads[index];
	}

	for (int history_index = *history_count - 1; history_index >= 0; history_index--) {
		int slot = (*head - 1 - history_index + history_size * 2) % history_size;
		double dot = 0.0;

		for (int index = 0; index < count; index++) {
			dot += s_history[slot * count + index] * direction[(size_t)index];
		}

		alphas[(size_t)history_index] = rho_history[slot] * dot;

		for (int index = 0; index < count; index++) {
			direction[(size_t)index] -= alphas[(size_t)history_index] *
				y_history[slot * count + index];
		}
	}

	if (*history_count > 0) {
		int slot = (*head - 1 + history_size * 2) % history_size;
		double yy = 0.0;
		double ys = 0.0;

		for (int index = 0; index < count; index++) {
			double y_value = y_history[slot * count + index];
			yy += y_value * y_value;
			ys += y_value * s_history[slot * count + index];
		}

		double gamma = ys / yy;

		for (int index = 0; index < count; index++) {
			direction[(size_t)index] *= gamma;
		}
	}

	for (int history_index = 0; history_index < *history_count; history_index++) {
		int slot = (*head - *history_count + history_index + history_size * 2) % history_size;
		double dot = 0.0;

		for (int index = 0; index < count; index++) {
			dot += y_history[slot * count + index] * direction[(size_t)index];
		}

		double beta = rho_history[slot] * dot;

		for (int index = 0; index < count; index++) {
			direction[(size_t)index] +=
				(alphas[(size_t)history_index] - beta) * s_history[slot * count + index];
		}
	}

	double effective_learning_rate = learning_rate;

	if (line_search != 0) {
		double f0 = 0.0;
		double slope = 0.0;
		double c1_value = c1 == 0.0 ? 1e-4 : c1;

		for (int index = 0; index < count; index++) {
			f0 += grads[index] * grads[index];
			slope -= grads[index] * direction[(size_t)index];
		}

		for (int search_index = 0; search_index < 50; search_index++) {
			double decrease = f0 - c1_value * effective_learning_rate * slope;

			if (decrease > 0.0) break;

			effective_learning_rate *= 0.5;

			if (effective_learning_rate < 1e-10) break;
		}

		if (effective_learning_rate < 1e-10) {
			effective_learning_rate = 1e-10;
		}
	}

	for (int index = 0; index < count; index++) {
		out[index] = params[index] - effective_learning_rate * direction[(size_t)index];
	}

	return 0;
}

} // extern "C"
