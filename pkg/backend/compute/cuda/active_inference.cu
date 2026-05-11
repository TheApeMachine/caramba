#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include "active_inference.h"

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) return -1; \
} while(0)

// ---------------------------------------------------------------------------
// free_energy_kernel: computes element contribution mu^2 + exp(ls) - ls - 1
// and reduces to partial sums.
// ---------------------------------------------------------------------------
__global__ void free_energy_kernel(
    const double* __restrict__ mu,
    const double* __restrict__ log_sigma,
    double* __restrict__ partials,
    int n)
{
    __shared__ double smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double val = 0.0;

    if (idx < n) {
        double m  = mu[idx];
        double ls = log_sigma[idx];
        val = m*m + exp(ls) - ls - 1.0;
    }

    smem[tid] = val;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    if (tid == 0) partials[blockIdx.x] = smem[0];
}

int cuda_ai_free_energy(const double* mu, const double* log_sigma, double* out, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double *dMu, *dLs, *dPartials;

    CUDA_CHECK(cudaMalloc(&dMu,      (size_t)n      * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dLs,      (size_t)n      * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dPartials, (size_t)blocks * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(dMu, mu,         (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dLs, log_sigma,  (size_t)n * sizeof(double), cudaMemcpyHostToDevice));

    free_energy_kernel<<<blocks, BLOCK_SIZE>>>(dMu, dLs, dPartials, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double* hPartials = (double*)malloc((size_t)blocks * sizeof(double));
    if (!hPartials) { cudaFree(dMu); cudaFree(dLs); cudaFree(dPartials); return -1; }

    CUDA_CHECK(cudaMemcpy(hPartials, dPartials, (size_t)blocks * sizeof(double), cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int i = 0; i < blocks; i++) sum += hPartials[i];
    free(hPartials);

    *out = 0.5 * sum;

    cudaFree(dMu); cudaFree(dLs); cudaFree(dPartials);
    return 0;
}

// ---------------------------------------------------------------------------
// belief_update_kernel: updates mu and log_sigma in parallel.
// out[0..n-1] = mu_new, out[n..2n-1] = log_sigma_new.
// ---------------------------------------------------------------------------
__global__ void belief_update_kernel(
    const double* __restrict__ mu,
    const double* __restrict__ log_sigma,
    const double* __restrict__ pred_err,
    double* __restrict__ out,
    double lr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    out[idx]     = mu[idx]        - lr * (mu[idx] + pred_err[idx]);
    out[idx + n] = log_sigma[idx] - lr * (exp(log_sigma[idx]) - 1.0);
}

int cuda_ai_belief_update(
    const double* mu, const double* log_sigma,
    const double* pred_err, double lr,
    double* out, int n)
{
    double *dMu, *dLs, *dPe, *dOut;
    size_t sz = (size_t)n * sizeof(double);

    CUDA_CHECK(cudaMalloc(&dMu,  sz));
    CUDA_CHECK(cudaMalloc(&dLs,  sz));
    CUDA_CHECK(cudaMalloc(&dPe,  sz));
    CUDA_CHECK(cudaMalloc(&dOut, 2*sz));

    CUDA_CHECK(cudaMemcpy(dMu, mu,         sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dLs, log_sigma,  sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPe, pred_err,   sz, cudaMemcpyHostToDevice));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    belief_update_kernel<<<blocks, BLOCK_SIZE>>>(dMu, dLs, dPe, dOut, lr, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out, dOut, 2*sz, cudaMemcpyDeviceToHost));

    cudaFree(dMu); cudaFree(dLs); cudaFree(dPe); cudaFree(dOut);
    return 0;
}

// ---------------------------------------------------------------------------
// precision_weight_kernel: out[i] = err[i] * exp(log_prec[i])
// ---------------------------------------------------------------------------
__global__ void precision_weight_kernel(
    const double* __restrict__ err,
    const double* __restrict__ log_prec,
    double* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    out[idx] = err[idx] * exp(log_prec[idx]);
}

int cuda_ai_precision_weight(
    const double* err, const double* log_prec, double* out, int n)
{
    double *dErr, *dLp, *dOut;
    size_t sz = (size_t)n * sizeof(double);

    CUDA_CHECK(cudaMalloc(&dErr, sz));
    CUDA_CHECK(cudaMalloc(&dLp,  sz));
    CUDA_CHECK(cudaMalloc(&dOut, sz));

    CUDA_CHECK(cudaMemcpy(dErr, err,      sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dLp,  log_prec, sz, cudaMemcpyHostToDevice));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    precision_weight_kernel<<<blocks, BLOCK_SIZE>>>(dErr, dLp, dOut, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out, dOut, sz, cudaMemcpyDeviceToHost));

    cudaFree(dErr); cudaFree(dLp); cudaFree(dOut);
    return 0;
}

// ---------------------------------------------------------------------------
// expected_free_energy_kernel:
// G[k] = -sum_i q[i,k]*ln(q[i,k]+eps)  for each outcome k.
// Each thread computes one k.
// ---------------------------------------------------------------------------
__global__ void expected_free_energy_kernel(
    const double* __restrict__ q_outcomes,
    double* __restrict__ out,
    int n, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= K) return;

    const double eps = 1e-12;
    double g = 0.0;

    for (int i = 0; i < n; i++) {
        double q = q_outcomes[i * K + k];
        g -= q * log(q + eps);
    }

    out[k] = g;
}

int cuda_ai_expected_free_energy(
    const double* q_outcomes, double* out, int n, int K)
{
    double *dQ, *dOut;
    size_t szQ   = (size_t)n * K * sizeof(double);
    size_t szOut = (size_t)K     * sizeof(double);

    CUDA_CHECK(cudaMalloc(&dQ,   szQ));
    CUDA_CHECK(cudaMalloc(&dOut, szOut));

    CUDA_CHECK(cudaMemcpy(dQ, q_outcomes, szQ, cudaMemcpyHostToDevice));

    int blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    expected_free_energy_kernel<<<blocks, BLOCK_SIZE>>>(dQ, dOut, n, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out, dOut, szOut, cudaMemcpyDeviceToHost));

    cudaFree(dQ); cudaFree(dOut);
    return 0;
}
