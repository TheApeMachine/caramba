#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "active_inference.h"

#define BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// free_energy_kernel: computes element contribution mu^2 + exp(ls) - ls - 1
// and reduces to partial sums.
// ---------------------------------------------------------------------------
__global__ void free_energy_kernel(
    const double* __restrict__ mu,
    const double* __restrict__ log_sigma,
    double* __restrict__ out_sum,
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

    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out_sum, smem[0]);
}

int cuda_ai_free_energy(const double* mu, const double* log_sigma, double* out, int n) {
    if (n < 0) return -1;

    if (n == 0) {
        *out = 0.0;
        return 0;
    }

    double *dMu = NULL;
    double *dLs = NULL;
    double *dSum = NULL;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaError_t err;

    err = cudaMalloc((void**)&dMu, (size_t)n * sizeof(double));
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc((void**)&dLs, (size_t)n * sizeof(double));
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc((void**)&dSum, sizeof(double));
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemset(dSum, 0, sizeof(double));
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(dMu, mu, (size_t)n * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(dLs, log_sigma, (size_t)n * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    free_energy_kernel<<<blocks, BLOCK_SIZE>>>(dMu, dLs, dSum, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup;

    {
        double sum = 0.0;
        err = cudaMemcpy(&sum, dSum, sizeof(double), cudaMemcpyDeviceToHost);
        if (err == cudaSuccess) *out = 0.5 * sum;
    }

cleanup:
    if (dMu) cudaFree(dMu);
    if (dLs) cudaFree(dLs);
    if (dSum) cudaFree(dSum);
    return (err == cudaSuccess) ? 0 : -1;
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
    if (n < 0) return -1;

    if (n == 0) {
        return 0;
    }

    double *dMu = NULL;
    double *dLs = NULL;
    double *dPe = NULL;
    double *dOut = NULL;
    size_t sz = (size_t)n * sizeof(double);
    cudaError_t err;

    err = cudaMalloc((void**)&dMu, sz);

    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc((void**)&dLs, sz);

    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc((void**)&dPe, sz);

    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc((void**)&dOut, 2 * sz);

    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(dMu, mu, sz, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(dLs, log_sigma, sz, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(dPe, pred_err, sz, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) goto cleanup;

    {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        belief_update_kernel<<<blocks, BLOCK_SIZE>>>(dMu, dLs, dPe, dOut, lr, n);
    }

    err = cudaGetLastError();

    if (err != cudaSuccess) goto cleanup;

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(out, dOut, 2 * sz, cudaMemcpyDeviceToHost);

cleanup:
    cudaFree(dMu);
    cudaFree(dLs);
    cudaFree(dPe);
    cudaFree(dOut);
    return (err == cudaSuccess) ? 0 : -1;
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
    if (n < 0) return -1;

    if (n == 0) {
        return 0;
    }

    double *dErr = NULL;
    double *dLp = NULL;
    double *dOut = NULL;
    size_t sz = (size_t)n * sizeof(double);
    cudaError_t errC = cudaSuccess;

    errC = cudaMalloc((void**)&dErr, sz);

    if (errC != cudaSuccess) goto cleanup;

    errC = cudaMalloc((void**)&dLp, sz);

    if (errC != cudaSuccess) goto cleanup;

    errC = cudaMalloc((void**)&dOut, sz);

    if (errC != cudaSuccess) goto cleanup;

    errC = cudaMemcpy(dErr, err, sz, cudaMemcpyHostToDevice);

    if (errC != cudaSuccess) goto cleanup;

    errC = cudaMemcpy(dLp, log_prec, sz, cudaMemcpyHostToDevice);

    if (errC != cudaSuccess) goto cleanup;

    {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        precision_weight_kernel<<<blocks, BLOCK_SIZE>>>(dErr, dLp, dOut, n);
    }

    errC = cudaGetLastError();

    if (errC != cudaSuccess) goto cleanup;

    errC = cudaDeviceSynchronize();

    if (errC != cudaSuccess) goto cleanup;

    errC = cudaMemcpy(out, dOut, sz, cudaMemcpyDeviceToHost);

cleanup:
    cudaFree(dErr);
    cudaFree(dLp);
    cudaFree(dOut);
    return (errC == cudaSuccess) ? 0 : -1;
}

// ---------------------------------------------------------------------------
// expected_free_energy_kernel:
// G[k] = -sum_i q[i,k]*ln(q[i,k]+eps). One CUDA block per outcome k;
// threads stride over i and parallel-reduce in shared memory (same pattern as
// free_energy_kernel).
// ---------------------------------------------------------------------------
__global__ void expected_free_energy_kernel(
    const double* __restrict__ q_outcomes,
    double* __restrict__ out,
    int n, int K)
{
    int k = blockIdx.x;

    if (k >= K) return;

    const double eps = 1e-12;
    __shared__ double smem[BLOCK_SIZE];
    int tid = threadIdx.x;

    double val = 0.0;

    for (int i = tid; i < n; i += blockDim.x) {
        double q = q_outcomes[i * K + k];
        val -= q * log(q + eps);
    }

    smem[tid] = val;
    __syncthreads();

    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    if (tid == 0) out[k] = smem[0];
}

int cuda_ai_expected_free_energy(
    const double* q_outcomes, double* out, int n, int K)
{
    if (n < 0 || K < 0) return -1;

    if (K == 0) {
        return 0;
    }

    if (n == 0) {
        for (int kk = 0; kk < K; kk++) out[kk] = 0.0;
        return 0;
    }

    size_t nzu = (size_t)n;
    size_t kzu = (size_t)K;

    if (nzu * kzu / kzu != nzu) {
        return -1;
    }

    if (nzu * kzu > SIZE_MAX / sizeof(double)) {
        return -1;
    }

    double *dQ = NULL;
    double *dOut = NULL;
    size_t szQ   = nzu * kzu * sizeof(double);
    size_t szOut = kzu * sizeof(double);
    cudaError_t err;

    err = cudaMalloc((void**)&dQ, szQ);

    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc((void**)&dOut, szOut);

    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(dQ, q_outcomes, szQ, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) goto cleanup;

    expected_free_energy_kernel<<<K, BLOCK_SIZE>>>(dQ, dOut, n, K);
    err = cudaGetLastError();

    if (err != cudaSuccess) goto cleanup;

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(out, dOut, szOut, cudaMemcpyDeviceToHost);

cleanup:
    cudaFree(dQ);
    cudaFree(dOut);
    return (err == cudaSuccess) ? 0 : -1;
}
