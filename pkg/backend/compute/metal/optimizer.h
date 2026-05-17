#ifndef METAL_OPTIMIZER_H
#define METAL_OPTIMIZER_H

#ifdef __cplusplus
extern "C" {
#endif

int metal_optimizer_init(const char *metallib_path);
const char *metal_optimizer_last_error(void);
int metal_optimizer_zero_tensor(void *buffer, int count);
int metal_optimizer_copy_tensor(const void *src, void *dst, int count);

int metal_optimizer_adam(
	double *out, double *moment, double *variance,
	const double *params, const double *grads, int count,
	double beta1, double beta2, double learning_rate, double eps
);
int metal_optimizer_adamw(
	double *out, double *moment, double *variance,
	const double *params, const double *grads, int count,
	double beta1, double beta2, double learning_rate, double eps,
	double weight_decay_step
);
int metal_optimizer_adamax(
	double *out, double *moment, double *infinity_norm,
	const double *params, const double *grads, int count,
	double beta1, double beta2, double learning_rate, double eps
);
int metal_optimizer_sgd(
	double *out, double *velocity,
	const double *params, const double *grads, int count,
	double learning_rate, double weight_decay, double momentum, int nesterov
);
int metal_optimizer_lion(
	double *out, double *moment,
	const double *params, const double *grads, int count,
	double learning_rate, double beta1, double beta2, double weight_decay
);
int metal_optimizer_rmsprop(
	double *out, double *square_average, double *momentum_buffer,
	double *grad_average, const double *params, const double *grads,
	int count, double learning_rate, double alpha, double eps,
	double momentum, double weight_decay, int centered
);
int metal_optimizer_hebbian(
	double *out, const double *params, const double *grads, int count,
	double learning_rate, double max_norm
);
int metal_optimizer_lars(
	double *out, double *velocity,
	const double *params, const double *grads, int count,
	double learning_rate, double eta, double momentum,
	double weight_decay, double eps
);
int metal_optimizer_lamb(
	double *out, double *moment, double *variance,
	const double *params, const double *grads, int count,
	double learning_rate, double beta1, double beta2, double eps,
	double weight_decay, double bias_correction1_inv,
	double bias_correction2_inv
);
int metal_optimizer_adagrad(
	double *out, double *accumulator,
	const double *params, const double *grads, int count,
	double learning_rate, double eps, double weight_decay
);
int metal_optimizer_adadelta(
	double *out, double *grad_average, double *delta_average,
	const double *params, const double *grads, int count,
	double rho, double eps, double weight_decay
);
int metal_optimizer_lbfgs(
	double *out, double *s_history, double *y_history, double *rho_history,
	int *head, int *history_count, const double *params,
	const double *grads, const double *previous_params,
	const double *previous_grads, int has_previous, int count,
	int history_size, double learning_rate, int line_search, double c1
);
int metal_optimizer_adam_tensor(
	const void *params, const void *grads, void *moment,
	void *variance, void *out, int count,
	double beta1, double beta2, double learning_rate, double eps
);
int metal_optimizer_adamw_tensor(
	const void *params, const void *grads, void *moment,
	void *variance, void *out, int count,
	double beta1, double beta2, double learning_rate, double eps,
	double weight_decay_step
);
int metal_optimizer_adamax_tensor(
	const void *params, const void *grads, void *moment,
	void *infinity_norm, void *out, int count,
	double beta1, double beta2, double learning_rate, double eps
);
int metal_optimizer_sgd_tensor(
	const void *params, const void *grads, void *velocity,
	void *out, int count,
	double learning_rate, double weight_decay, double momentum, int nesterov
);
int metal_optimizer_lion_tensor(
	const void *params, const void *grads, void *moment,
	void *out, int count,
	double learning_rate, double beta1, double beta2, double weight_decay
);
int metal_optimizer_rmsprop_tensor(
	const void *params, const void *grads, void *square_average,
	void *momentum_buffer, void *grad_average, void *out, int count,
	double learning_rate, double alpha, double eps,
	double momentum, double weight_decay, int centered
);
int metal_optimizer_hebbian_tensor(
	const void *params, const void *grads, void *out, int count,
	double learning_rate, double max_norm
);
int metal_optimizer_lars_tensor(
	const void *params, const void *grads, void *velocity,
	void *out, int count,
	double learning_rate, double eta, double momentum,
	double weight_decay, double eps
);
int metal_optimizer_lamb_tensor(
	const void *params, const void *grads, void *moment,
	void *variance, void *out, int count,
	double learning_rate, double beta1, double beta2, double eps,
	double weight_decay, double bias_correction1_inv,
	double bias_correction2_inv
);
int metal_optimizer_adagrad_tensor(
	const void *params, const void *grads, void *accumulator,
	void *out, int count,
	double learning_rate, double eps, double weight_decay
);
int metal_optimizer_adadelta_tensor(
	const void *params, const void *grads, void *grad_average,
	void *delta_average, void *out, int count,
	double rho, double eps, double weight_decay
);
/*
 * metal_optimizer_lbfgs_tensor -- L-BFGS update step over Metal-resident buffers.
 *
 * Buffer size requirements (the caller must allocate exactly these counts):
 *   state_history -- history_size * count floats (ring buffer of s-vectors).
 *   grad_history  -- history_size * count floats (ring buffer of y-vectors).
 *   rho_history   -- history_size floats (1 / (y . s) per history slot).
 *   head          -- 1 element; current ring-buffer insert index.
 *   history_count -- 1 element; number of populated slots (clipped to history_size).
 *   previous_params, previous_grads -- count elements each; last accepted params/grads.
 *
 * Semantics: `head` advances modulo history_size on each accepted curvature pair;
 * `history_count` grows toward history_size and saturates. `rho_history[head]` is
 * written alongside slot `head` of state_history / grad_history. Callers must hold
 * these buffers stable across calls so the two-loop recursion sees a valid history.
 */
int metal_optimizer_lbfgs_tensor(
	const void *params, const void *grads, void *state_history,
	void *grad_history, void *rho_history, void *head,
	void *history_count, void *previous_params,
	void *previous_grads, void *out, int has_previous, int count,
	int history_size, double learning_rate, int line_search, double c1
);

#ifdef __cplusplus
}
#endif

#endif
