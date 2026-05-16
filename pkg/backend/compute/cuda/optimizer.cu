#include <cuda_runtime.h>
#include <math.h>
#include "optimizer.h"

#define BLOCK_SIZE 256

#define CUDA_OPT_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) return -1; \
} while (0)

static int optimizer_alloc_copy(const double* host, double** device, int count) {
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMalloc((void**)device, bytes));
    CUDA_OPT_CHECK(cudaMemcpy(*device, host, bytes, cudaMemcpyHostToDevice));

    return 0;
}

static int optimizer_copy_free(double* host, double* device, int count) {
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
    cudaFree(device);

    return 0;
}

__global__ void optimizer_adam_kernel(
    double* out, double* moment, double* variance,
    const double* params, const double* grads, int count,
    double beta1, double beta2, double learning_rate, double eps
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double grad = grads[index];
    double next_moment = beta1 * moment[index] + (1.0 - beta1) * grad;
    double next_variance = beta2 * variance[index] + (1.0 - beta2) * grad * grad;

    moment[index] = next_moment;
    variance[index] = next_variance;
    out[index] = params[index] - learning_rate * next_moment / (sqrt(next_variance) + eps);
}

__global__ void optimizer_adamw_kernel(
    double* out, double* moment, double* variance,
    const double* params, const double* grads, int count,
    double beta1, double beta2, double learning_rate, double eps,
    double weight_decay_step
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double grad = grads[index];
    double next_moment = beta1 * moment[index] + (1.0 - beta1) * grad;
    double next_variance = beta2 * variance[index] + (1.0 - beta2) * grad * grad;
    double decayed_param = params[index] - weight_decay_step * params[index];

    moment[index] = next_moment;
    variance[index] = next_variance;
    out[index] = decayed_param - learning_rate * next_moment / (sqrt(next_variance) + eps);
}

__global__ void optimizer_adamax_kernel(
    double* out, double* moment, double* infinity_norm,
    const double* params, const double* grads, int count,
    double beta1, double beta2, double learning_rate, double eps
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double grad = grads[index];
    double next_moment = beta1 * moment[index] + (1.0 - beta1) * grad;
    double next_norm = fmax(beta2 * infinity_norm[index], fabs(grad));

    moment[index] = next_moment;
    infinity_norm[index] = next_norm;
    out[index] = params[index] - learning_rate * next_moment / (next_norm + eps);
}

__global__ void optimizer_sgd_kernel(
    double* out, double* velocity,
    const double* params, const double* grads, int count,
    double learning_rate, double weight_decay, double momentum,
    int nesterov
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double grad = grads[index] + weight_decay * params[index];

    if (momentum == 0.0) {
        out[index] = params[index] - learning_rate * grad;
        return;
    }

    double next_velocity = momentum * velocity[index] + grad;
    velocity[index] = next_velocity;
    double update = nesterov ? grad + momentum * next_velocity : next_velocity;
    out[index] = params[index] - learning_rate * update;
}

__global__ void optimizer_lion_kernel(
    double* out, double* moment,
    const double* params, const double* grads, int count,
    double learning_rate, double beta1, double beta2, double weight_decay
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double blended = beta1 * moment[index] + (1.0 - beta1) * grads[index];
    double sign = (blended > 0.0) ? 1.0 : (blended < 0.0 ? -1.0 : 0.0);
    out[index] = params[index] - learning_rate * (sign + weight_decay * params[index]);
    moment[index] = beta2 * moment[index] + (1.0 - beta2) * grads[index];
}

__global__ void optimizer_rmsprop_kernel(
    double* out, double* square_average, double* momentum_buffer,
    double* grad_average, const double* params, const double* grads,
    int count, double learning_rate, double alpha, double eps,
    double momentum, double weight_decay, int centered
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double grad = grads[index] + weight_decay * params[index];
    double next_square = alpha * square_average[index] + (1.0 - alpha) * grad * grad;
    square_average[index] = next_square;

    double average = next_square;
    if (centered) {
        double next_grad_average = alpha * grad_average[index] + (1.0 - alpha) * grad;
        grad_average[index] = next_grad_average;
        average -= next_grad_average * next_grad_average;
    }

    double update = grad / (sqrt(average) + eps);
    if (momentum != 0.0) {
        update = momentum * momentum_buffer[index] + update;
        momentum_buffer[index] = update;
    }

    out[index] = params[index] - learning_rate * update;
}

__global__ void optimizer_hebbian_kernel(
    double* out, double* norm_square, const double* params, const double* grads,
    int count, double learning_rate
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double value = params[index] + learning_rate * grads[index];
    out[index] = value;
    atomicAdd(norm_square, value * value);
}

__global__ void optimizer_hebbian_scale_kernel(double* out, int count, double scale) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) out[index] *= scale;
}

__global__ void optimizer_lars_kernel(
    double* out, double* velocity,
    const double* params, const double* grads, int count,
    double learning_rate, double eta, double momentum,
    double weight_decay, double eps
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double param_norm_square = 0.0;
    double grad_norm_square = 0.0;

    for (int norm_index = 0; norm_index < count; norm_index++) {
        param_norm_square += params[norm_index] * params[norm_index];
        grad_norm_square += grads[norm_index] * grads[norm_index];
    }

    double param_norm = sqrt(param_norm_square);
    double grad_norm = sqrt(grad_norm_square);
    double local_learning_rate = learning_rate;

    if (param_norm > 0.0 && grad_norm > 0.0) {
        local_learning_rate = eta * param_norm / (grad_norm + weight_decay * param_norm + eps);
    }

    double grad = grads[index] + weight_decay * params[index];
    double next_velocity = momentum * velocity[index] + local_learning_rate * grad;
    velocity[index] = next_velocity;
    out[index] = params[index] - next_velocity;
}

__global__ void optimizer_lamb_kernel(
    double* out, double* moment, double* variance,
    const double* params, const double* grads, int count,
    double learning_rate, double beta1, double beta2, double eps,
    double weight_decay, double bias_correction1_inv,
    double bias_correction2_inv
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double param_norm_square = 0.0;
    double update_norm_square = 0.0;

    for (int norm_index = 0; norm_index < count; norm_index++) {
        double next_moment = beta1 * moment[norm_index] + (1.0 - beta1) * grads[norm_index];
        double next_variance = beta2 * variance[norm_index] +
            (1.0 - beta2) * grads[norm_index] * grads[norm_index];
        double update = next_moment * bias_correction1_inv /
            (sqrt(next_variance * bias_correction2_inv) + eps) +
            weight_decay * params[norm_index];
        param_norm_square += params[norm_index] * params[norm_index];
        update_norm_square += update * update;
    }

    double grad = grads[index];
    double next_moment = beta1 * moment[index] + (1.0 - beta1) * grad;
    double next_variance = beta2 * variance[index] + (1.0 - beta2) * grad * grad;
    double update = next_moment * bias_correction1_inv /
        (sqrt(next_variance * bias_correction2_inv) + eps) + weight_decay * params[index];
    double param_norm = sqrt(param_norm_square);
    double update_norm = sqrt(update_norm_square);
    double ratio = learning_rate;

    if (param_norm > 0.0 && update_norm > 0.0) {
        ratio = learning_rate * param_norm / update_norm;
    }

    moment[index] = next_moment;
    variance[index] = next_variance;
    out[index] = params[index] - ratio * update;
}

__global__ void optimizer_adagrad_kernel(
    double* out, double* accumulator,
    const double* params, const double* grads, int count,
    double learning_rate, double eps, double weight_decay
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double grad = grads[index] + weight_decay * params[index];
    accumulator[index] += grad * grad;
    out[index] = params[index] - learning_rate * grad / (sqrt(accumulator[index]) + eps);
}

__global__ void optimizer_adadelta_kernel(
    double* out, double* grad_average, double* delta_average,
    const double* params, const double* grads, int count,
    double rho, double eps, double weight_decay
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double grad = grads[index] + weight_decay * params[index];
    grad_average[index] = rho * grad_average[index] + (1.0 - rho) * grad * grad;
    double update = sqrt(delta_average[index] + eps) / sqrt(grad_average[index] + eps) * grad;
    out[index] = params[index] - update;
    delta_average[index] = rho * delta_average[index] + (1.0 - rho) * update * update;
}

__global__ void optimizer_lbfgs_kernel(
    double* out, double* s_history, double* y_history, double* rho_history,
    double* direction, double* alphas,
    int* head, int* history_count, const double* params,
    const double* grads, const double* previous_params,
    const double* previous_grads, int has_previous, int count,
    int history_size, double learning_rate, int line_search, double c1
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    if (has_previous && history_size > 0) {
        int slot = (*head) % history_size;
        double curvature = 0.0;

        for (int index = 0; index < count; index++) {
            double s = params[index] - previous_params[index];
            double y = grads[index] - previous_grads[index];
            s_history[slot * count + index] = s;
            y_history[slot * count + index] = y;
            curvature += y * s;
        }

        if (curvature > 1e-10) {
            rho_history[slot] = 1.0 / curvature;
            *head += 1;
            if (*history_count < history_size) *history_count += 1;
        }
    }

    for (int index = 0; index < count; index++) {
        direction[index] = grads[index];
    }

    for (int history_index = *history_count - 1; history_index >= 0; history_index--) {
        int slot = (*head - 1 - history_index + history_size * 2) % history_size;
        double dot = 0.0;

        for (int index = 0; index < count; index++) {
            dot += s_history[slot * count + index] * direction[index];
        }

        alphas[history_index] = rho_history[slot] * dot;

        for (int index = 0; index < count; index++) {
            direction[index] -= alphas[history_index] * y_history[slot * count + index];
        }
    }

    if (*history_count > 0) {
        int slot = (*head - 1 + history_size * 2) % history_size;
        double yy = 0.0;
        double ys = 0.0;

        for (int index = 0; index < count; index++) {
            yy += y_history[slot * count + index] * y_history[slot * count + index];
            ys += y_history[slot * count + index] * s_history[slot * count + index];
        }

        double gamma = ys / yy;

        for (int index = 0; index < count; index++) {
            direction[index] *= gamma;
        }
    }

    for (int history_index = 0; history_index < *history_count; history_index++) {
        int slot = (*head - *history_count + history_index + history_size * 2) % history_size;
        double dot = 0.0;

        for (int index = 0; index < count; index++) {
            dot += y_history[slot * count + index] * direction[index];
        }

        double beta = rho_history[slot] * dot;

        for (int index = 0; index < count; index++) {
            direction[index] += (alphas[history_index] - beta) *
                s_history[slot * count + index];
        }
    }

    double effective_learning_rate = learning_rate;

    if (line_search) {
        double f0 = 0.0;
        double slope = 0.0;
        double c1_value = (c1 == 0.0) ? 1e-4 : c1;

        for (int index = 0; index < count; index++) {
            f0 += grads[index] * grads[index];
            slope -= grads[index] * direction[index];
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
        out[index] = params[index] - effective_learning_rate * direction[index];
    }
}

extern "C" {

int cuda_optimizer_adam(
    double* out, double* moment, double* variance,
    const double* params, const double* grads, int count,
    double beta1, double beta2, double learning_rate, double eps
) {
    double *dOut, *dMoment, *dVariance, *dParams, *dGrads;
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    if (optimizer_alloc_copy(moment, &dMoment, count)) return -1;
    if (optimizer_alloc_copy(variance, &dVariance, count)) return -1;
    if (optimizer_alloc_copy(params, &dParams, count)) return -1;
    if (optimizer_alloc_copy(grads, &dGrads, count)) return -1;
    optimizer_adam_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dOut, dMoment, dVariance, dParams, dGrads, count,
        beta1, beta2, learning_rate, eps
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    optimizer_copy_free(out, dOut, count);
    optimizer_copy_free(moment, dMoment, count);
    optimizer_copy_free(variance, dVariance, count);
    cudaFree(dParams); cudaFree(dGrads);
    return 0;
}

int cuda_optimizer_adamw(
    double* out, double* moment, double* variance,
    const double* params, const double* grads, int count,
    double beta1, double beta2, double learning_rate, double eps,
    double weight_decay_step
) {
    double *dOut, *dMoment, *dVariance, *dParams, *dGrads;
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    if (optimizer_alloc_copy(moment, &dMoment, count)) return -1;
    if (optimizer_alloc_copy(variance, &dVariance, count)) return -1;
    if (optimizer_alloc_copy(params, &dParams, count)) return -1;
    if (optimizer_alloc_copy(grads, &dGrads, count)) return -1;
    optimizer_adamw_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dOut, dMoment, dVariance, dParams, dGrads, count,
        beta1, beta2, learning_rate, eps, weight_decay_step
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    optimizer_copy_free(out, dOut, count);
    optimizer_copy_free(moment, dMoment, count);
    optimizer_copy_free(variance, dVariance, count);
    cudaFree(dParams); cudaFree(dGrads);
    return 0;
}

int cuda_optimizer_adamax(
    double* out, double* moment, double* infinity_norm,
    const double* params, const double* grads, int count,
    double beta1, double beta2, double learning_rate, double eps
) {
    double *dOut, *dMoment, *dNorm, *dParams, *dGrads;
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    if (optimizer_alloc_copy(moment, &dMoment, count)) return -1;
    if (optimizer_alloc_copy(infinity_norm, &dNorm, count)) return -1;
    if (optimizer_alloc_copy(params, &dParams, count)) return -1;
    if (optimizer_alloc_copy(grads, &dGrads, count)) return -1;
    optimizer_adamax_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dOut, dMoment, dNorm, dParams, dGrads, count, beta1, beta2, learning_rate, eps
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    optimizer_copy_free(out, dOut, count);
    optimizer_copy_free(moment, dMoment, count);
    optimizer_copy_free(infinity_norm, dNorm, count);
    cudaFree(dParams); cudaFree(dGrads);
    return 0;
}

int cuda_optimizer_sgd(
    double* out, double* velocity,
    const double* params, const double* grads, int count,
    double learning_rate, double weight_decay, double momentum,
    int nesterov
) {
    double *dOut, *dVelocity, *dParams, *dGrads;
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    if (optimizer_alloc_copy(velocity, &dVelocity, count)) return -1;
    if (optimizer_alloc_copy(params, &dParams, count)) return -1;
    if (optimizer_alloc_copy(grads, &dGrads, count)) return -1;
    optimizer_sgd_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dOut, dVelocity, dParams, dGrads, count,
        learning_rate, weight_decay, momentum, nesterov
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    optimizer_copy_free(out, dOut, count);
    optimizer_copy_free(velocity, dVelocity, count);
    cudaFree(dParams); cudaFree(dGrads);
    return 0;
}

int cuda_optimizer_lion(
    double* out, double* moment,
    const double* params, const double* grads, int count,
    double learning_rate, double beta1, double beta2, double weight_decay
) {
    double *dOut, *dMoment, *dParams, *dGrads;
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    if (optimizer_alloc_copy(moment, &dMoment, count)) return -1;
    if (optimizer_alloc_copy(params, &dParams, count)) return -1;
    if (optimizer_alloc_copy(grads, &dGrads, count)) return -1;
    optimizer_lion_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dOut, dMoment, dParams, dGrads, count, learning_rate, beta1, beta2, weight_decay
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    optimizer_copy_free(out, dOut, count);
    optimizer_copy_free(moment, dMoment, count);
    cudaFree(dParams); cudaFree(dGrads);
    return 0;
}

int cuda_optimizer_rmsprop(
    double* out, double* square_average, double* momentum_buffer,
    double* grad_average, const double* params, const double* grads,
    int count, double learning_rate, double alpha, double eps,
    double momentum, double weight_decay, int centered
) {
    double *dOut, *dSquare, *dMomentum, *dGradAverage, *dParams, *dGrads;
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    if (optimizer_alloc_copy(square_average, &dSquare, count)) return -1;
    if (optimizer_alloc_copy(momentum_buffer, &dMomentum, count)) return -1;
    if (optimizer_alloc_copy(grad_average, &dGradAverage, count)) return -1;
    if (optimizer_alloc_copy(params, &dParams, count)) return -1;
    if (optimizer_alloc_copy(grads, &dGrads, count)) return -1;
    optimizer_rmsprop_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dOut, dSquare, dMomentum, dGradAverage, dParams, dGrads, count,
        learning_rate, alpha, eps, momentum, weight_decay, centered
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    optimizer_copy_free(out, dOut, count);
    optimizer_copy_free(square_average, dSquare, count);
    optimizer_copy_free(momentum_buffer, dMomentum, count);
    optimizer_copy_free(grad_average, dGradAverage, count);
    cudaFree(dParams); cudaFree(dGrads);
    return 0;
}

int cuda_optimizer_hebbian(
    double* out, const double* params, const double* grads, int count,
    double learning_rate, double max_norm
) {
    double *dOut, *dParams, *dGrads, *dNorm;
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    CUDA_OPT_CHECK(cudaMalloc((void**)&dNorm, sizeof(double)));
    CUDA_OPT_CHECK(cudaMemset(dNorm, 0, sizeof(double)));
    if (optimizer_alloc_copy(params, &dParams, count)) return -1;
    if (optimizer_alloc_copy(grads, &dGrads, count)) return -1;
    optimizer_hebbian_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dOut, dNorm, dParams, dGrads, count, learning_rate
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    if (max_norm > 0.0) {
        double norm_square = 0.0;
        CUDA_OPT_CHECK(cudaMemcpy(&norm_square, dNorm, sizeof(double), cudaMemcpyDeviceToHost));
        double norm = sqrt(norm_square);
        if (norm > max_norm) {
            optimizer_hebbian_scale_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                dOut, count, max_norm / norm
            );
            CUDA_OPT_CHECK(cudaGetLastError());
        }
    }
    optimizer_copy_free(out, dOut, count);
    cudaFree(dParams); cudaFree(dGrads); cudaFree(dNorm);
    return 0;
}

int cuda_optimizer_lars(
    double* out, double* velocity,
    const double* params, const double* grads, int count,
    double learning_rate, double eta, double momentum,
    double weight_decay, double eps
) {
    double *dOut, *dVelocity, *dParams, *dGrads;
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    if (optimizer_alloc_copy(velocity, &dVelocity, count)) return -1;
    if (optimizer_alloc_copy(params, &dParams, count)) return -1;
    if (optimizer_alloc_copy(grads, &dGrads, count)) return -1;
    optimizer_lars_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dOut, dVelocity, dParams, dGrads, count,
        learning_rate, eta, momentum, weight_decay, eps
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    optimizer_copy_free(out, dOut, count);
    optimizer_copy_free(velocity, dVelocity, count);
    cudaFree(dParams); cudaFree(dGrads);
    return 0;
}

int cuda_optimizer_lamb(
    double* out, double* moment, double* variance,
    const double* params, const double* grads, int count,
    double learning_rate, double beta1, double beta2, double eps,
    double weight_decay, double bias_correction1_inv,
    double bias_correction2_inv
) {
    double *dOut, *dMoment, *dVariance, *dParams, *dGrads;
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    if (optimizer_alloc_copy(moment, &dMoment, count)) return -1;
    if (optimizer_alloc_copy(variance, &dVariance, count)) return -1;
    if (optimizer_alloc_copy(params, &dParams, count)) return -1;
    if (optimizer_alloc_copy(grads, &dGrads, count)) return -1;
    optimizer_lamb_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dOut, dMoment, dVariance, dParams, dGrads, count,
        learning_rate, beta1, beta2, eps, weight_decay,
        bias_correction1_inv, bias_correction2_inv
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    optimizer_copy_free(out, dOut, count);
    optimizer_copy_free(moment, dMoment, count);
    optimizer_copy_free(variance, dVariance, count);
    cudaFree(dParams); cudaFree(dGrads);
    return 0;
}

int cuda_optimizer_adagrad(
    double* out, double* accumulator,
    const double* params, const double* grads, int count,
    double learning_rate, double eps, double weight_decay
) {
    double *dOut, *dAccumulator, *dParams, *dGrads;
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    if (optimizer_alloc_copy(accumulator, &dAccumulator, count)) return -1;
    if (optimizer_alloc_copy(params, &dParams, count)) return -1;
    if (optimizer_alloc_copy(grads, &dGrads, count)) return -1;
    optimizer_adagrad_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dOut, dAccumulator, dParams, dGrads, count, learning_rate, eps, weight_decay
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    optimizer_copy_free(out, dOut, count);
    optimizer_copy_free(accumulator, dAccumulator, count);
    cudaFree(dParams); cudaFree(dGrads);
    return 0;
}

int cuda_optimizer_adadelta(
    double* out, double* grad_average, double* delta_average,
    const double* params, const double* grads, int count,
    double rho, double eps, double weight_decay
) {
    double *dOut, *dGradAverage, *dDeltaAverage, *dParams, *dGrads;
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    if (optimizer_alloc_copy(grad_average, &dGradAverage, count)) return -1;
    if (optimizer_alloc_copy(delta_average, &dDeltaAverage, count)) return -1;
    if (optimizer_alloc_copy(params, &dParams, count)) return -1;
    if (optimizer_alloc_copy(grads, &dGrads, count)) return -1;
    optimizer_adadelta_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dOut, dGradAverage, dDeltaAverage, dParams, dGrads, count,
        rho, eps, weight_decay
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    optimizer_copy_free(out, dOut, count);
    optimizer_copy_free(grad_average, dGradAverage, count);
    optimizer_copy_free(delta_average, dDeltaAverage, count);
    cudaFree(dParams); cudaFree(dGrads);
    return 0;
}

int cuda_optimizer_lbfgs(
    double* out, double* s_history, double* y_history, double* rho_history,
    int* head, int* history_count, const double* params,
    const double* grads, const double* previous_params,
    const double* previous_grads, int has_previous, int count,
    int history_size, double learning_rate, int line_search, double c1
) {
    double *dOut, *dS, *dY, *dRho, *dDirection, *dAlphas;
    double *dParams, *dGrads, *dPrevParams, *dPrevGrads;
    int *dHead, *dCount;
    int history_elements = count * history_size;
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    CUDA_OPT_CHECK(cudaMalloc((void**)&dDirection, bytes));
    CUDA_OPT_CHECK(cudaMalloc((void**)&dAlphas, (size_t)history_size * sizeof(double)));
    if (optimizer_alloc_copy(s_history, &dS, history_elements)) return -1;
    if (optimizer_alloc_copy(y_history, &dY, history_elements)) return -1;
    if (optimizer_alloc_copy(rho_history, &dRho, history_size)) return -1;
    if (optimizer_alloc_copy(params, &dParams, count)) return -1;
    if (optimizer_alloc_copy(grads, &dGrads, count)) return -1;
    if (optimizer_alloc_copy(previous_params, &dPrevParams, count)) return -1;
    if (optimizer_alloc_copy(previous_grads, &dPrevGrads, count)) return -1;
    CUDA_OPT_CHECK(cudaMalloc((void**)&dHead, sizeof(int)));
    CUDA_OPT_CHECK(cudaMalloc((void**)&dCount, sizeof(int)));
    CUDA_OPT_CHECK(cudaMemcpy(dHead, head, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OPT_CHECK(cudaMemcpy(dCount, history_count, sizeof(int), cudaMemcpyHostToDevice));
    optimizer_lbfgs_kernel<<<1, 1>>>(
        dOut, dS, dY, dRho, dDirection, dAlphas, dHead, dCount, dParams, dGrads,
        dPrevParams, dPrevGrads, has_previous, count, history_size, learning_rate,
        line_search, c1
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    optimizer_copy_free(out, dOut, count);
    optimizer_copy_free(s_history, dS, history_elements);
    optimizer_copy_free(y_history, dY, history_elements);
    optimizer_copy_free(rho_history, dRho, history_size);
    CUDA_OPT_CHECK(cudaMemcpy(head, dHead, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_OPT_CHECK(cudaMemcpy(history_count, dCount, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(dParams); cudaFree(dGrads); cudaFree(dPrevParams); cudaFree(dPrevGrads);
    cudaFree(dHead); cudaFree(dCount); cudaFree(dDirection); cudaFree(dAlphas);
    return 0;
}

} // extern "C"
