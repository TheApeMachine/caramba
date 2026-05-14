#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include "training.h"

#define BLOCK_SIZE 256

#define TRAIN_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) return -1; \
} while(0)

static int training_alloc_copy(const void* host, void** device, size_t bytes) {
    *device = NULL;
    if (cudaMalloc(device, bytes) != cudaSuccess) {
        *device = NULL;
        return -1;
    }
    if (cudaMemcpy(*device, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*device);
        *device = NULL;
        return -1;
    }
    return 0;
}

__global__ void mse_loss_kernel(
    const double* predictions, const double* targets, double* sum, int n)
{
    extern __shared__ double shared[];
    int thread_index = threadIdx.x;
    int index = blockIdx.x * blockDim.x + thread_index;
    double value = 0.0;

    if (index < n) {
        double diff = predictions[index] - targets[index];
        value = diff * diff;
    }

    shared[thread_index] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) shared[thread_index] += shared[thread_index + stride];
        __syncthreads();
    }

    if (thread_index == 0) atomicAdd(sum, shared[0]);
}

__global__ void mse_grad_kernel(
    const double* predictions, const double* targets, double* out, int n)
{
    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (index >= n) return;
    out[index] = 2.0 * (predictions[index] - targets[index]) / (double)n;
}

__global__ void max_reduce_kernel(const double* input, double* out, int n)
{
    extern __shared__ double shared[];
    int thread_index = threadIdx.x;
    double max_value = -1.0e300;

    for (int index = thread_index; index < n; index += blockDim.x) {
        double value = input[index];
        max_value = max_value > value ? max_value : value;
    }

    shared[thread_index] = max_value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            double value = shared[thread_index + stride];
            shared[thread_index] = shared[thread_index] > value ? shared[thread_index] : value;
        }
        __syncthreads();
    }

    if (thread_index == 0) out[0] = shared[0];
}

__global__ void softmax_sum_kernel(const double* logits, double* sum, double max_value, int n)
{
    extern __shared__ double shared[];
    int thread_index = threadIdx.x;
    int index = blockIdx.x * blockDim.x + thread_index;
    double value = 0.0;

    if (index < n) value = exp(logits[index] - max_value);

    shared[thread_index] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) shared[thread_index] += shared[thread_index + stride];
        __syncthreads();
    }

    if (thread_index == 0) atomicAdd(sum, shared[0]);
}

__global__ void cross_entropy_loss_kernel(
    const double* logits,
    const double* targets,
    double* out,
    double max_value,
    double sum_value,
    int n)
{
    extern __shared__ double shared[];
    int thread_index = threadIdx.x;
    int index = blockIdx.x * blockDim.x + thread_index;
    double value = 0.0;

    if (index < n) {
        double probability = exp(logits[index] - max_value) / sum_value;
        value = -log(probability + 1.0e-9) * targets[index];
    }

    shared[thread_index] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) shared[thread_index] += shared[thread_index + stride];
        __syncthreads();
    }

    if (thread_index == 0) atomicAdd(out, shared[0]);
}

__global__ void cross_entropy_grad_kernel(
    const double* logits,
    const double* targets,
    double* out,
    double max_value,
    double sum_value,
    int n)
{
    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (index >= n) return;
    out[index] = exp(logits[index] - max_value) / sum_value - targets[index];
}

__global__ void accuracy_kernel(const double* predictions, const double* targets, double* out, int n)
{
    __shared__ double pred_values[BLOCK_SIZE];
    __shared__ double target_values[BLOCK_SIZE];
    __shared__ int pred_indices[BLOCK_SIZE];
    __shared__ int target_indices[BLOCK_SIZE];
    int thread_index = threadIdx.x;
    double pred_best = -1.0e300;
    double target_best = -1.0e300;
    int pred_index = 0;
    int target_index = 0;

    for (int index = thread_index; index < n; index += blockDim.x) {
        if (predictions[index] > pred_best) {
            pred_best = predictions[index];
            pred_index = index;
        }
        if (targets[index] > target_best) {
            target_best = targets[index];
            target_index = index;
        }
    }

    pred_values[thread_index] = pred_best;
    target_values[thread_index] = target_best;
    pred_indices[thread_index] = pred_index;
    target_indices[thread_index] = target_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            if (pred_values[thread_index + stride] > pred_values[thread_index]) {
                pred_values[thread_index] = pred_values[thread_index + stride];
                pred_indices[thread_index] = pred_indices[thread_index + stride];
            }
            if (target_values[thread_index + stride] > target_values[thread_index]) {
                target_values[thread_index] = target_values[thread_index + stride];
                target_indices[thread_index] = target_indices[thread_index + stride];
            }
        }
        __syncthreads();
    }

    if (thread_index == 0) out[0] = pred_indices[0] == target_indices[0] ? 1.0 : 0.0;
}

__global__ void f1_counts_kernel(const double* predictions, const double* targets, double* out, int n)
{
    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (index >= n) return;

    bool predicted = predictions[index] >= 0.5;
    bool actual = targets[index] >= 0.5;

    if (predicted && actual) atomicAdd(&out[0], 1.0);
    if (predicted && !actual) atomicAdd(&out[1], 1.0);
    if (!predicted && actual) atomicAdd(&out[2], 1.0);
    if (!predicted && !actual) atomicAdd(&out[3], 1.0);
}

static int cross_entropy_common(
    const double* logits,
    const double* targets,
    double* out,
    int n,
    int gradient)
{
    double *dLogits = NULL, *dTargets = NULL, *dOut = NULL;
    double *dMax = NULL, *dSum = NULL;
    size_t bytes = (size_t)n * sizeof(double);
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double max_value = 0.0, sum_value = 0.0;

    if (training_alloc_copy(logits, (void**)&dLogits, bytes)) return -1;
    if (training_alloc_copy(targets, (void**)&dTargets, bytes)) goto fail;
    if (cudaMalloc((void**)&dMax, sizeof(double)) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&dSum, sizeof(double)) != cudaSuccess) goto fail;

    max_reduce_kernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(dLogits, dMax, n);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaMemcpy(&max_value, dMax, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;
    if (cudaMemset(dSum, 0, sizeof(double)) != cudaSuccess) goto fail;
    softmax_sum_kernel<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(dLogits, dSum, max_value, n);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaMemcpy(&sum_value, dSum, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    if (gradient) {
        if (cudaMalloc((void**)&dOut, bytes) != cudaSuccess) goto fail;
        cross_entropy_grad_kernel<<<blocks, BLOCK_SIZE>>>(
            dLogits, dTargets, dOut, max_value, sum_value, n
        );
        if (cudaGetLastError() != cudaSuccess) goto fail;
        if (cudaMemcpy(out, dOut, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;
    } else {
        if (cudaMalloc((void**)&dOut, sizeof(double)) != cudaSuccess) goto fail;
        if (cudaMemset(dOut, 0, sizeof(double)) != cudaSuccess) goto fail;
        cross_entropy_loss_kernel<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
            dLogits, dTargets, dOut, max_value, sum_value, n
        );
        if (cudaGetLastError() != cudaSuccess) goto fail;
        if (cudaMemcpy(out, dOut, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;
    }

    cudaFree(dLogits); cudaFree(dTargets); cudaFree(dOut); cudaFree(dMax); cudaFree(dSum);
    return 0;
fail:
    cudaFree(dLogits); cudaFree(dTargets); cudaFree(dOut); cudaFree(dMax); cudaFree(dSum);
    return -1;
}

extern "C" {

int cuda_train_mse_loss(const double* predictions, const double* targets, double* out, size_t n) {
    if (!predictions || !targets || !out || n == 0 || n > INT_MAX) return -1;

    int count = (int)n;
    double *dPredictions = NULL, *dTargets = NULL, *dSum = NULL;
    size_t bytes = n * sizeof(double);
    int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (training_alloc_copy(predictions, (void**)&dPredictions, bytes)) return -1;
    if (training_alloc_copy(targets, (void**)&dTargets, bytes)) goto fail;
    if (cudaMalloc((void**)&dSum, sizeof(double)) != cudaSuccess) goto fail;
    if (cudaMemset(dSum, 0, sizeof(double)) != cudaSuccess) goto fail;

    mse_loss_kernel<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
        dPredictions, dTargets, dSum, count
    );
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaMemcpy(out, dSum, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;
    out[0] /= (double)count;

    cudaFree(dPredictions); cudaFree(dTargets); cudaFree(dSum);
    return 0;
fail:
    cudaFree(dPredictions); cudaFree(dTargets); cudaFree(dSum);
    return -1;
}

int cuda_train_cross_entropy_loss(const double* logits, const double* targets, double* out, size_t n) {
    if (!logits || !targets || !out || n == 0 || n > INT_MAX) return -1;
    return cross_entropy_common(logits, targets, out, (int)n, 0);
}

int cuda_train_mse_grad(const double* predictions, const double* targets, double* out, size_t n) {
    if (!predictions || !targets || !out || n == 0 || n > INT_MAX) return -1;

    int count = (int)n;
    double *dPredictions = NULL, *dTargets = NULL, *dOut = NULL;
    size_t bytes = n * sizeof(double);

    if (training_alloc_copy(predictions, (void**)&dPredictions, bytes)) return -1;
    if (training_alloc_copy(targets, (void**)&dTargets, bytes)) goto fail;
    if (cudaMalloc((void**)&dOut, bytes) != cudaSuccess) goto fail;

    mse_grad_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dPredictions, dTargets, dOut, count
    );
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaMemcpy(out, dOut, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(dPredictions); cudaFree(dTargets); cudaFree(dOut);
    return 0;
fail:
    cudaFree(dPredictions); cudaFree(dTargets); cudaFree(dOut);
    return -1;
}

int cuda_train_cross_entropy_grad(const double* logits, const double* targets, double* out, size_t n) {
    if (!logits || !targets || !out || n == 0 || n > INT_MAX) return -1;
    return cross_entropy_common(logits, targets, out, (int)n, 1);
}

int cuda_metric_accuracy(const double* predictions, const double* targets, double* out, size_t n) {
    if (!predictions || !targets || !out || n == 0 || n > INT_MAX) return -1;

    int count = (int)n;
    double *dPredictions = NULL, *dTargets = NULL, *dOut = NULL;
    size_t bytes = n * sizeof(double);

    if (training_alloc_copy(predictions, (void**)&dPredictions, bytes)) return -1;
    if (training_alloc_copy(targets, (void**)&dTargets, bytes)) goto fail;
    if (cudaMalloc((void**)&dOut, sizeof(double)) != cudaSuccess) goto fail;

    accuracy_kernel<<<1, BLOCK_SIZE>>>(dPredictions, dTargets, dOut, count);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaMemcpy(out, dOut, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(dPredictions); cudaFree(dTargets); cudaFree(dOut);
    return 0;
fail:
    cudaFree(dPredictions); cudaFree(dTargets); cudaFree(dOut);
    return -1;
}

int cuda_metric_f1_counts(const double* predictions, const double* targets, double* out, size_t n) {
    if (!predictions || !targets || !out || n == 0 || n > INT_MAX) return -1;

    int count = (int)n;
    double *dPredictions = NULL, *dTargets = NULL, *dOut = NULL;
    size_t bytes = n * sizeof(double);

    if (training_alloc_copy(predictions, (void**)&dPredictions, bytes)) return -1;
    if (training_alloc_copy(targets, (void**)&dTargets, bytes)) goto fail;
    if (cudaMalloc((void**)&dOut, 4 * sizeof(double)) != cudaSuccess) goto fail;
    if (cudaMemset(dOut, 0, 4 * sizeof(double)) != cudaSuccess) goto fail;

    f1_counts_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dPredictions, dTargets, dOut, count
    );
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaMemcpy(out, dOut, 4 * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(dPredictions); cudaFree(dTargets); cudaFree(dOut);
    return 0;
fail:
    cudaFree(dPredictions); cudaFree(dTargets); cudaFree(dOut);
    return -1;
}

} // extern "C"
