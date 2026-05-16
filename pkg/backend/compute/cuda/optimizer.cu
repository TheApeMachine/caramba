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

static int optimizer_groups(int count) {
    return count <= 0 ? 0 : (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
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

__global__ void optimizer_reduce_pair_kernel(
    const double2* input, double2* output, int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int local = threadIdx.x;
    __shared__ double left[BLOCK_SIZE];
    __shared__ double right[BLOCK_SIZE];

    double2 value = make_double2(0.0, 0.0);

    if (index < count) {
        value = input[index];
    }

    left[local] = value.x;
    right[local] = value.y;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (local < stride) {
            left[local] += left[local + stride];
            right[local] += right[local + stride];
        }

        __syncthreads();
    }

    if (local == 0) {
        output[blockIdx.x] = make_double2(left[0], right[0]);
    }
}

__global__ void optimizer_lars_norms_kernel(
    const double* params, const double* grads, double2* partials, int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int local = threadIdx.x;
    __shared__ double param_squares[BLOCK_SIZE];
    __shared__ double grad_squares[BLOCK_SIZE];

    double param_square = 0.0;
    double grad_square = 0.0;

    if (index < count) {
        double param = params[index];
        double grad = grads[index];
        param_square = param * param;
        grad_square = grad * grad;
    }

    param_squares[local] = param_square;
    grad_squares[local] = grad_square;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (local < stride) {
            param_squares[local] += param_squares[local + stride];
            grad_squares[local] += grad_squares[local + stride];
        }

        __syncthreads();
    }

    if (local == 0) {
        partials[blockIdx.x] = make_double2(param_squares[0], grad_squares[0]);
    }
}

__global__ void optimizer_lars_apply_kernel(
    double* out, double* velocity,
    const double* params, const double* grads, const double2* norms, int count,
    double learning_rate, double eta, double momentum,
    double weight_decay, double eps
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double param_norm = sqrt(norms[0].x);
    double grad_norm = sqrt(norms[0].y);
    double local_learning_rate = learning_rate;

    if (param_norm > 0.0 && grad_norm > 0.0) {
        local_learning_rate = eta * param_norm / (grad_norm + weight_decay * param_norm + eps);
    }

    double grad = grads[index] + weight_decay * params[index];
    double next_velocity = momentum * velocity[index] + local_learning_rate * grad;
    velocity[index] = next_velocity;
    out[index] = params[index] - next_velocity;
}

__global__ void optimizer_lamb_prepare_kernel(
    double* update, double* moment, double* variance,
    const double* params, const double* grads, double2* partials, int count,
    double beta1, double beta2, double eps, double weight_decay,
    double bias_correction1_inv, double bias_correction2_inv
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int local = threadIdx.x;
    __shared__ double param_squares[BLOCK_SIZE];
    __shared__ double update_squares[BLOCK_SIZE];

    double param_square = 0.0;
    double update_square = 0.0;

    if (index < count) {
        double grad = grads[index];
        double param = params[index];
        double next_moment = beta1 * moment[index] + (1.0 - beta1) * grad;
        double next_variance = beta2 * variance[index] + (1.0 - beta2) * grad * grad;
        double next_update = next_moment * bias_correction1_inv /
            (sqrt(next_variance * bias_correction2_inv) + eps) + weight_decay * param;

        moment[index] = next_moment;
        variance[index] = next_variance;
        update[index] = next_update;
        param_square = param * param;
        update_square = next_update * next_update;
    }

    param_squares[local] = param_square;
    update_squares[local] = update_square;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (local < stride) {
            param_squares[local] += param_squares[local + stride];
            update_squares[local] += update_squares[local + stride];
        }

        __syncthreads();
    }

    if (local == 0) {
        partials[blockIdx.x] = make_double2(param_squares[0], update_squares[0]);
    }
}

__global__ void optimizer_lamb_apply_kernel(
    double* out, const double* update, const double* params, const double2* norms,
    int count, double learning_rate
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double ratio = learning_rate;

    if (norms[0].x > 0.0 && norms[0].y > 0.0) {
        ratio = learning_rate * sqrt(norms[0].x) / sqrt(norms[0].y);
    }

    out[index] = params[index] - ratio * update[index];
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

__global__ void optimizer_lbfgs_history_delta_kernel(
    double* s_history, double* y_history, double2* partials,
    const double* params, const double* grads,
    const double* previous_params, const double* previous_grads,
    const int* head, int count, int history_size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int local = threadIdx.x;
    int slot = head[0] % history_size;
    __shared__ double curvature_values[BLOCK_SIZE];

    double curvature = 0.0;

    if (index < count) {
        double state_delta = params[index] - previous_params[index];
        double grad_delta = grads[index] - previous_grads[index];
        s_history[slot * count + index] = state_delta;
        y_history[slot * count + index] = grad_delta;
        curvature = grad_delta * state_delta;
    }

    curvature_values[local] = curvature;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (local < stride) {
            curvature_values[local] += curvature_values[local + stride];
        }

        __syncthreads();
    }

    if (local == 0) {
        partials[blockIdx.x] = make_double2(curvature_values[0], 0.0);
    }
}

__global__ void optimizer_lbfgs_accept_history_kernel(
    double* rho_history, int* head, int* history_count,
    const double2* curvature, int history_size
) {
    double value = curvature[0].x;

    if (value <= 1e-10) return;

    int slot = head[0] % history_size;
    rho_history[slot] = 1.0 / value;
    head[0] += 1;

    if (history_count[0] < history_size) {
        history_count[0] += 1;
    }
}

__global__ void optimizer_lbfgs_direction_init_kernel(
    double* direction, const double* grads, int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) direction[index] = grads[index];
}

__global__ void optimizer_lbfgs_dot_kernel(
    const double* left, const double* right, double2* partials, int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int local = threadIdx.x;
    __shared__ double values[BLOCK_SIZE];

    values[local] = index < count ? left[index] * right[index] : 0.0;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (local < stride) {
            values[local] += values[local + stride];
        }

        __syncthreads();
    }

    if (local == 0) {
        partials[blockIdx.x] = make_double2(values[0], 0.0);
    }
}

__global__ void optimizer_lbfgs_gamma_kernel(
    const double* y_slot, const double* s_slot, double2* partials, int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int local = threadIdx.x;
    __shared__ double yy_values[BLOCK_SIZE];
    __shared__ double ys_values[BLOCK_SIZE];

    double yy = 0.0;
    double ys = 0.0;

    if (index < count) {
        double y_value = y_slot[index];
        yy = y_value * y_value;
        ys = y_value * s_slot[index];
    }

    yy_values[local] = yy;
    ys_values[local] = ys;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (local < stride) {
            yy_values[local] += yy_values[local + stride];
            ys_values[local] += ys_values[local + stride];
        }

        __syncthreads();
    }

    if (local == 0) {
        partials[blockIdx.x] = make_double2(yy_values[0], ys_values[0]);
    }
}

__global__ void optimizer_lbfgs_store_alpha_kernel(
    double* alphas, const double* rho_history, const double2* dot,
    int history_index, int slot
) {
    alphas[history_index] = rho_history[slot] * dot[0].x;
}

__global__ void optimizer_lbfgs_reverse_apply_kernel(
    double* direction, const double* y_slot, const double* alphas,
    int history_index, int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    direction[index] -= alphas[history_index] * y_slot[index];
}

__global__ void optimizer_lbfgs_gamma_apply_kernel(
    double* direction, const double2* gamma_pair, int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double gamma = gamma_pair[0].x == 0.0 ? 1.0 : gamma_pair[0].y / gamma_pair[0].x;
    direction[index] *= gamma;
}

__global__ void optimizer_lbfgs_forward_apply_kernel(
    double* direction, const double* s_slot, const double* rho_history,
    const double* alphas, const double2* dot,
    int history_index, int slot, int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double beta = rho_history[slot] * dot[0].x;
    direction[index] += (alphas[history_index] - beta) * s_slot[index];
}

__global__ void optimizer_lbfgs_line_search_kernel(
    const double* grads, const double* direction, double2* partials, int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int local = threadIdx.x;
    __shared__ double f_values[BLOCK_SIZE];
    __shared__ double slope_values[BLOCK_SIZE];

    double f_value = 0.0;
    double slope_value = 0.0;

    if (index < count) {
        double grad = grads[index];
        f_value = grad * grad;
        slope_value = grad * direction[index];
    }

    f_values[local] = f_value;
    slope_values[local] = slope_value;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (local < stride) {
            f_values[local] += f_values[local + stride];
            slope_values[local] += slope_values[local + stride];
        }

        __syncthreads();
    }

    if (local == 0) {
        partials[blockIdx.x] = make_double2(f_values[0], slope_values[0]);
    }
}

__global__ void optimizer_lbfgs_finalize_kernel(
    double* out, const double* params, const double* direction,
    const double2* line_metrics, int count,
    double learning_rate, int line_search, double c1
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    double effective_learning_rate = learning_rate;

    if (line_search) {
        double f0 = line_metrics[0].x;
        double slope = -line_metrics[0].y;
        double c1_value = c1 == 0.0 ? 1e-4 : c1;

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

    out[index] = params[index] - effective_learning_rate * direction[index];
}

static int optimizer_reduce_pair(double2* partials, int partial_count, double2** reduced) {
    double2* current = partials;
    int current_count = partial_count;

    while (current_count > 1) {
        int next_count = optimizer_groups(current_count);
        double2* next = nullptr;
        CUDA_OPT_CHECK(cudaMalloc((void**)&next, (size_t)next_count * sizeof(double2)));

        optimizer_reduce_pair_kernel<<<next_count, BLOCK_SIZE>>>(current, next, current_count);
        CUDA_OPT_CHECK(cudaGetLastError());

        if (current != partials) {
            cudaFree(current);
        }

        current = next;
        current_count = next_count;
    }

    *reduced = current;
    return 0;
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
    double2 *dPartials, *dNorms;
    size_t bytes = (size_t)count * sizeof(double);
    int group_count = optimizer_groups(count);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    CUDA_OPT_CHECK(cudaMalloc((void**)&dPartials, (size_t)group_count * sizeof(double2)));
    if (optimizer_alloc_copy(velocity, &dVelocity, count)) return -1;
    if (optimizer_alloc_copy(params, &dParams, count)) return -1;
    if (optimizer_alloc_copy(grads, &dGrads, count)) return -1;

    optimizer_lars_norms_kernel<<<group_count, BLOCK_SIZE>>>(dParams, dGrads, dPartials, count);
    CUDA_OPT_CHECK(cudaGetLastError());
    if (optimizer_reduce_pair(dPartials, group_count, &dNorms)) return -1;

    optimizer_lars_apply_kernel<<<group_count, BLOCK_SIZE>>>(
        dOut, dVelocity, dParams, dGrads, dNorms, count,
        learning_rate, eta, momentum, weight_decay, eps
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    optimizer_copy_free(out, dOut, count);
    optimizer_copy_free(velocity, dVelocity, count);
    if (dNorms != dPartials) cudaFree(dNorms);
    cudaFree(dPartials);
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
    double2 *dPartials, *dNorms;
    size_t bytes = (size_t)count * sizeof(double);
    int group_count = optimizer_groups(count);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    CUDA_OPT_CHECK(cudaMalloc((void**)&dPartials, (size_t)group_count * sizeof(double2)));
    if (optimizer_alloc_copy(moment, &dMoment, count)) return -1;
    if (optimizer_alloc_copy(variance, &dVariance, count)) return -1;
    if (optimizer_alloc_copy(params, &dParams, count)) return -1;
    if (optimizer_alloc_copy(grads, &dGrads, count)) return -1;

    optimizer_lamb_prepare_kernel<<<group_count, BLOCK_SIZE>>>(
        dOut, dMoment, dVariance, dParams, dGrads, dPartials, count,
        beta1, beta2, eps, weight_decay, bias_correction1_inv, bias_correction2_inv
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    if (optimizer_reduce_pair(dPartials, group_count, &dNorms)) return -1;

    optimizer_lamb_apply_kernel<<<group_count, BLOCK_SIZE>>>(
        dOut, dOut, dParams, dNorms, count, learning_rate
    );
    CUDA_OPT_CHECK(cudaGetLastError());
    optimizer_copy_free(out, dOut, count);
    optimizer_copy_free(moment, dMoment, count);
    optimizer_copy_free(variance, dVariance, count);
    if (dNorms != dPartials) cudaFree(dNorms);
    cudaFree(dPartials);
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
    if (count <= 0) return 0;
    if (history_size <= 0) return -1;

    double *dOut, *dS, *dY, *dRho, *dDirection, *dAlphas;
    double *dParams, *dGrads, *dPrevParams, *dPrevGrads;
    double2 *dPartials;
    int *dHead, *dCount;
    int history_elements = count * history_size;
    int group_count = optimizer_groups(count);
    size_t bytes = (size_t)count * sizeof(double);
    CUDA_OPT_CHECK(cudaMalloc((void**)&dOut, bytes));
    CUDA_OPT_CHECK(cudaMalloc((void**)&dDirection, bytes));
    CUDA_OPT_CHECK(cudaMalloc((void**)&dAlphas, (size_t)history_size * sizeof(double)));
    CUDA_OPT_CHECK(cudaMalloc((void**)&dPartials, (size_t)group_count * sizeof(double2)));
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

    if (has_previous) {
        double2* dCurvature = nullptr;
        optimizer_lbfgs_history_delta_kernel<<<group_count, BLOCK_SIZE>>>(
            dS, dY, dPartials, dParams, dGrads, dPrevParams, dPrevGrads,
            dHead, count, history_size
        );
        CUDA_OPT_CHECK(cudaGetLastError());
        if (optimizer_reduce_pair(dPartials, group_count, &dCurvature)) return -1;
        optimizer_lbfgs_accept_history_kernel<<<1, 1>>>(dRho, dHead, dCount, dCurvature, history_size);
        CUDA_OPT_CHECK(cudaGetLastError());
        if (dCurvature != dPartials) cudaFree(dCurvature);
        CUDA_OPT_CHECK(cudaMemcpy(head, dHead, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_OPT_CHECK(cudaMemcpy(history_count, dCount, sizeof(int), cudaMemcpyDeviceToHost));
    }

    optimizer_lbfgs_direction_init_kernel<<<group_count, BLOCK_SIZE>>>(dDirection, dGrads, count);
    CUDA_OPT_CHECK(cudaGetLastError());

    for (int history_index = *history_count - 1; history_index >= 0; history_index--) {
        int slot = (*head - 1 - history_index + history_size * 2) % history_size;
        double2* dDot = nullptr;

        optimizer_lbfgs_dot_kernel<<<group_count, BLOCK_SIZE>>>(
            dS + slot * count, dDirection, dPartials, count
        );
        CUDA_OPT_CHECK(cudaGetLastError());
        if (optimizer_reduce_pair(dPartials, group_count, &dDot)) return -1;

        optimizer_lbfgs_store_alpha_kernel<<<1, 1>>>(dAlphas, dRho, dDot, history_index, slot);
        CUDA_OPT_CHECK(cudaGetLastError());
        optimizer_lbfgs_reverse_apply_kernel<<<group_count, BLOCK_SIZE>>>(
            dDirection, dY + slot * count, dAlphas, history_index, count
        );
        CUDA_OPT_CHECK(cudaGetLastError());

        if (dDot != dPartials) cudaFree(dDot);
    }

    if (*history_count > 0) {
        int slot = (*head - 1 + history_size * 2) % history_size;
        double2* dGamma = nullptr;

        optimizer_lbfgs_gamma_kernel<<<group_count, BLOCK_SIZE>>>(
            dY + slot * count, dS + slot * count, dPartials, count
        );
        CUDA_OPT_CHECK(cudaGetLastError());
        if (optimizer_reduce_pair(dPartials, group_count, &dGamma)) return -1;

        optimizer_lbfgs_gamma_apply_kernel<<<group_count, BLOCK_SIZE>>>(dDirection, dGamma, count);
        CUDA_OPT_CHECK(cudaGetLastError());

        if (dGamma != dPartials) cudaFree(dGamma);
    }

    for (int history_index = 0; history_index < *history_count; history_index++) {
        int slot = (*head - *history_count + history_index + history_size * 2) % history_size;
        double2* dDot = nullptr;

        optimizer_lbfgs_dot_kernel<<<group_count, BLOCK_SIZE>>>(
            dY + slot * count, dDirection, dPartials, count
        );
        CUDA_OPT_CHECK(cudaGetLastError());
        if (optimizer_reduce_pair(dPartials, group_count, &dDot)) return -1;

        optimizer_lbfgs_forward_apply_kernel<<<group_count, BLOCK_SIZE>>>(
            dDirection, dS + slot * count, dRho, dAlphas, dDot,
            history_index, slot, count
        );
        CUDA_OPT_CHECK(cudaGetLastError());

        if (dDot != dPartials) cudaFree(dDot);
    }

    double2* dLineMetrics = dPartials;

    if (line_search) {
        optimizer_lbfgs_line_search_kernel<<<group_count, BLOCK_SIZE>>>(
            dGrads, dDirection, dPartials, count
        );
        CUDA_OPT_CHECK(cudaGetLastError());
        if (optimizer_reduce_pair(dPartials, group_count, &dLineMetrics)) return -1;
    }

    optimizer_lbfgs_finalize_kernel<<<group_count, BLOCK_SIZE>>>(
        dOut, dParams, dDirection, dLineMetrics, count, learning_rate, line_search, c1
    );
    CUDA_OPT_CHECK(cudaGetLastError());

    if (line_search && dLineMetrics != dPartials) cudaFree(dLineMetrics);
    optimizer_copy_free(out, dOut, count);
    optimizer_copy_free(s_history, dS, history_elements);
    optimizer_copy_free(y_history, dY, history_elements);
    optimizer_copy_free(rho_history, dRho, history_size);
    CUDA_OPT_CHECK(cudaMemcpy(head, dHead, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_OPT_CHECK(cudaMemcpy(history_count, dCount, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(dParams); cudaFree(dGrads); cudaFree(dPrevParams); cudaFree(dPrevGrads);
    cudaFree(dHead); cudaFree(dCount); cudaFree(dDirection); cudaFree(dAlphas); cudaFree(dPartials);
    return 0;
}

} // extern "C"
