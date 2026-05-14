#ifndef CUDA_OPTIMIZER_H
#define CUDA_OPTIMIZER_H

#ifdef __cplusplus
extern "C" {
#endif

int cuda_optimizer_adam(
    double* out, double* moment, double* variance,
    const double* params, const double* grads, int count,
    double beta1, double beta2, double learning_rate, double eps
);

int cuda_optimizer_adamw(
    double* out, double* moment, double* variance,
    const double* params, const double* grads, int count,
    double beta1, double beta2, double learning_rate, double eps,
    double weight_decay_step
);

int cuda_optimizer_adamax(
    double* out, double* moment, double* infinity_norm,
    const double* params, const double* grads, int count,
    double beta1, double beta2, double learning_rate, double eps
);

int cuda_optimizer_sgd(
    double* out, double* velocity,
    const double* params, const double* grads, int count,
    double learning_rate, double weight_decay, double momentum,
    int nesterov
);

int cuda_optimizer_lion(
    double* out, double* moment,
    const double* params, const double* grads, int count,
    double learning_rate, double beta1, double beta2, double weight_decay
);

int cuda_optimizer_rmsprop(
    double* out, double* square_average, double* momentum_buffer,
    double* grad_average, const double* params, const double* grads,
    int count, double learning_rate, double alpha, double eps,
    double momentum, double weight_decay, int centered
);

int cuda_optimizer_hebbian(
    double* out, const double* params, const double* grads, int count,
    double learning_rate, double max_norm
);

int cuda_optimizer_lars(
    double* out, double* velocity,
    const double* params, const double* grads, int count,
    double learning_rate, double eta, double momentum,
    double weight_decay, double eps
);

int cuda_optimizer_lamb(
    double* out, double* moment, double* variance,
    const double* params, const double* grads, int count,
    double learning_rate, double beta1, double beta2, double eps,
    double weight_decay, double bias_correction1_inv,
    double bias_correction2_inv
);

int cuda_optimizer_adagrad(
    double* out, double* accumulator,
    const double* params, const double* grads, int count,
    double learning_rate, double eps, double weight_decay
);

int cuda_optimizer_adadelta(
    double* out, double* grad_average, double* delta_average,
    const double* params, const double* grads, int count,
    double rho, double eps, double weight_decay
);

int cuda_optimizer_lbfgs(
    double* out, double* s_history, double* y_history, double* rho_history,
    int* head, int* history_count, const double* params,
    const double* grads, const double* previous_params,
    const double* previous_grads, int has_previous, int count,
    int history_size, double learning_rate, int line_search, double c1
);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_OPTIMIZER_H */
