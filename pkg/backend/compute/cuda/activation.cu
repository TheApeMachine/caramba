#include <cuda_runtime.h>
#include <math.h>
#include "activation.h"

// ---------------------------------------------------------------------------
// Device kernels — all use double (float64) directly.
// ---------------------------------------------------------------------------

__global__ void relu_kernel(const double* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = src[i];
    dst[i] = x > 0.0 ? x : 0.0;
}

__global__ void leaky_relu_kernel(const double* src, double* dst, double alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = src[i];
    dst[i] = x >= 0.0 ? x : alpha * x;
}

static constexpr double kGeluSqrt2OverPi = 0.7978845608028654;
static constexpr double kGeluCoeff       = 0.044715;

__global__ void gelu_kernel(const double* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Tanh-approx GELU matches CPU/Metal/XLA semantics.
    double x = src[i];
    double x3 = x * x * x;
    double inner = kGeluSqrt2OverPi * (x + kGeluCoeff * x3);
    dst[i] = 0.5 * x * (1.0 + tanh(inner));
}

__global__ void tanh_kernel(const double* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = tanh(src[i]);
}

__global__ void sigmoid_kernel(const double* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = 1.0 / (1.0 + exp(-src[i]));
}

__global__ void swish_kernel(const double* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = src[i];
    dst[i] = x / (1.0 + exp(-x));
}

static constexpr double kSELUAlpha = 1.6732632423543772;
static constexpr double kSELUScale = 1.0507009873554805;

__global__ void selu_kernel(const double* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = src[i];
    dst[i] = x > 0.0 ? kSELUScale * x : kSELUScale * kSELUAlpha * (exp(x) - 1.0);
}

__global__ void swiglu_kernel(const double* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // src[0..n-1] = gates, src[n..2n-1] = values
    double gate  = src[i];
    double value = src[n + i];
    double sig   = 1.0 / (1.0 + exp(-gate));
    dst[i] = sig * value;
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

static const int BLOCK = 256;

static inline int blocks(int n) { return (n + BLOCK - 1) / BLOCK; }

static int synchronize_launch() {
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) return -1;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    return 0;
}

// ---------------------------------------------------------------------------
// C linkage wrappers — host memory in, host memory out.
// All device memory management is internal.
// ---------------------------------------------------------------------------

extern "C" {

int cuda_relu(const double* src, double* dst, int n) {
    double *d_src = NULL, *d_dst = NULL;
    size_t bytes = (size_t)n * sizeof(double);

    if (cudaMalloc(&d_src, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, bytes) != cudaSuccess) { cudaFree(d_src); return -1; }

    if (cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    relu_kernel<<<blocks(n), BLOCK>>>(d_src, d_dst, n);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;

    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src);
    cudaFree(d_dst);
    return -1;
}

int cuda_leaky_relu(const double* src, double* dst, double alpha, int n) {
    double *d_src = NULL, *d_dst = NULL;
    size_t bytes = (size_t)n * sizeof(double);

    if (cudaMalloc(&d_src, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, bytes) != cudaSuccess) { cudaFree(d_src); return -1; }

    if (cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    leaky_relu_kernel<<<blocks(n), BLOCK>>>(d_src, d_dst, alpha, n);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;

    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src);
    cudaFree(d_dst);
    return -1;
}

int cuda_gelu(const double* src, double* dst, int n) {
    double *d_src = NULL, *d_dst = NULL;
    size_t bytes = (size_t)n * sizeof(double);

    if (cudaMalloc(&d_src, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, bytes) != cudaSuccess) { cudaFree(d_src); return -1; }

    if (cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    gelu_kernel<<<blocks(n), BLOCK>>>(d_src, d_dst, n);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;

    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src);
    cudaFree(d_dst);
    return -1;
}

int cuda_tanh(const double* src, double* dst, int n) {
    double *d_src = NULL, *d_dst = NULL;
    size_t bytes = (size_t)n * sizeof(double);

    if (cudaMalloc(&d_src, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, bytes) != cudaSuccess) { cudaFree(d_src); return -1; }

    if (cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    tanh_kernel<<<blocks(n), BLOCK>>>(d_src, d_dst, n);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;

    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src);
    cudaFree(d_dst);
    return -1;
}

int cuda_sigmoid(const double* src, double* dst, int n) {
    double *d_src = NULL, *d_dst = NULL;
    size_t bytes = (size_t)n * sizeof(double);

    if (cudaMalloc(&d_src, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, bytes) != cudaSuccess) { cudaFree(d_src); return -1; }

    if (cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    sigmoid_kernel<<<blocks(n), BLOCK>>>(d_src, d_dst, n);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;

    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src);
    cudaFree(d_dst);
    return -1;
}

int cuda_swish(const double* src, double* dst, int n) {
    double *d_src = NULL, *d_dst = NULL;
    size_t bytes = (size_t)n * sizeof(double);

    if (cudaMalloc(&d_src, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, bytes) != cudaSuccess) { cudaFree(d_src); return -1; }

    if (cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    swish_kernel<<<blocks(n), BLOCK>>>(d_src, d_dst, n);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;

    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src);
    cudaFree(d_dst);
    return -1;
}

int cuda_selu(const double* src, double* dst, int n) {
    double *d_src = NULL, *d_dst = NULL;
    size_t bytes = (size_t)n * sizeof(double);

    if (cudaMalloc(&d_src, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, bytes) != cudaSuccess) { cudaFree(d_src); return -1; }

    if (cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    selu_kernel<<<blocks(n), BLOCK>>>(d_src, d_dst, n);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;

    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src);
    cudaFree(d_dst);
    return -1;
}

int cuda_swiglu(const double* src, double* dst, int n) {
    // src has 2*n elements; dst has n elements.
    double *d_src = NULL, *d_dst = NULL;
    size_t src_bytes = (size_t)(2 * n) * sizeof(double);
    size_t dst_bytes = (size_t)n * sizeof(double);

    if (cudaMalloc(&d_src, src_bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, dst_bytes) != cudaSuccess) { cudaFree(d_src); return -1; }

    if (cudaMemcpy(d_src, src, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    swiglu_kernel<<<blocks(n), BLOCK>>>(d_src, d_dst, n);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;

    if (cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src);
    cudaFree(d_dst);
    return -1;
}

int cuda_relu_device(const double* src, double* dst, int n) {
    if (n == 0) return 0;
    if (n < 0) return -1;
    if (!src || !dst) return -1;

    relu_kernel<<<blocks(n), BLOCK>>>(src, dst, n);
    return synchronize_launch();
}

int cuda_leaky_relu_device(const double* src, double* dst, double alpha, int n) {
    if (n == 0) return 0;
    if (n < 0) return -1;
    if (!src || !dst) return -1;

    leaky_relu_kernel<<<blocks(n), BLOCK>>>(src, dst, alpha, n);
    return synchronize_launch();
}

int cuda_gelu_device(const double* src, double* dst, int n) {
    if (n == 0) return 0;
    if (n < 0) return -1;
    if (!src || !dst) return -1;

    gelu_kernel<<<blocks(n), BLOCK>>>(src, dst, n);
    return synchronize_launch();
}

int cuda_tanh_device(const double* src, double* dst, int n) {
    if (n == 0) return 0;
    if (n < 0) return -1;
    if (!src || !dst) return -1;

    tanh_kernel<<<blocks(n), BLOCK>>>(src, dst, n);
    return synchronize_launch();
}

int cuda_sigmoid_device(const double* src, double* dst, int n) {
    if (n == 0) return 0;
    if (n < 0) return -1;
    if (!src || !dst) return -1;

    sigmoid_kernel<<<blocks(n), BLOCK>>>(src, dst, n);
    return synchronize_launch();
}

int cuda_swish_device(const double* src, double* dst, int n) {
    if (n == 0) return 0;
    if (n < 0) return -1;
    if (!src || !dst) return -1;

    swish_kernel<<<blocks(n), BLOCK>>>(src, dst, n);
    return synchronize_launch();
}

int cuda_selu_device(const double* src, double* dst, int n) {
    if (n == 0) return 0;
    if (n < 0) return -1;
    if (!src || !dst) return -1;

    selu_kernel<<<blocks(n), BLOCK>>>(src, dst, n);
    return synchronize_launch();
}

// cuda_swiglu_device: src must hold at least 2*n doubles (gates [0..n-1], values [n..2n-1]);
// see swiglu_kernel indexing src[i] and src[n+i].
int cuda_swiglu_device(const double* src, double* dst, int n) {
    if (n == 0) return 0;
    if (n < 0) return -1;
    if (!src || !dst) return -1;

    swiglu_kernel<<<blocks(n), BLOCK>>>(src, dst, n);
    return synchronize_launch();
}

} // extern "C"
