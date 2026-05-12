#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "hawkes.h"

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) return -1; \
} while(0)

static int alloc_copy(const void* h, void** d, size_t bytes) {
    if (cudaMalloc(d, bytes) != cudaSuccess) return -1;
    if (cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*d); return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Intensity kernel: each thread handles one process k.
// Computes lambda_k(t) = mu_k + alpha_k * sum_i exp(-beta_k*(t-times[i]))
// ---------------------------------------------------------------------------
__global__ void intensity_kernel(
    const double* __restrict__ times,
    const double* __restrict__ alpha,
    const double* __restrict__ beta,
    const double* __restrict__ mu,
    double* __restrict__ out,
    double t,
    int K, int T
) {
    int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (k >= K) return;

    double bk = beta[k];
    double sum = 0.0;

    for (int i = 0; i < T; i++) {
        double dt = t - times[i];
        if (dt <= 0.0) break;
        sum += exp(-bk * dt);
    }

    out[k] = mu[k] + alpha[k] * sum;
}

// ---------------------------------------------------------------------------
// Kernel matrix: K[i,j] = alpha * exp(-beta*(t_j - t_i)) for j > i.
// One thread per (i,j) pair.
// ---------------------------------------------------------------------------
__global__ void kernel_matrix_kernel(
    const double* __restrict__ times,
    double alpha, double beta,
    double* __restrict__ out,
    int T
) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = T * T;
    if (idx >= total) return;

    int row = idx / T;
    int col = idx % T;

    if (col <= row) {
        out[idx] = 0.0;
        return;
    }

    double dt = times[col] - times[row];
    out[idx] = alpha * exp(-beta * dt);
}

// ---------------------------------------------------------------------------
// Log-likelihood: sum log(lambda_i) computed in parallel, then subtract integral.
// ---------------------------------------------------------------------------
__global__ void log_sum_kernel(
    const double* __restrict__ intensities,
    double* __restrict__ out_sum,
    int T
) {
    extern __shared__ double smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    double val = 0.0;
    if (gid < T) {
        double lam = intensities[gid];
        if (lam > 0.0) val = log(lam);
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

__global__ void hawkes_fill_neg_one_kernel(double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -1.0;
}

// ---------------------------------------------------------------------------
// Simulate kernel: one thread per process — Ogata thinning.
// Uses curand-style LCG for each thread.
// ---------------------------------------------------------------------------
__global__ void simulate_kernel(
    const double* __restrict__ mu,
    const double* __restrict__ alpha,
    const double* __restrict__ beta,
    double T_max,
    double* __restrict__ out,
    int K, int maxSteps
) {
    int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (k >= K) return;

    // Simple LCG random number generator (thread-local seed).
    uint64_t seed = (uint64_t)(k + 1) * 6364136223846793005ULL + 1442695040888963407ULL;

    double* kevents = out + k * maxSteps;
    for (int s = 0; s < maxSteps; s++) kevents[s] = -1.0;

    double t = 0.0;
    int count = 0;
    double muk   = mu[k];
    double alphak = alpha[k];
    double betak  = beta[k];

    while (t < T_max && count < maxSteps) {
        // intensity upper bound
        double lstar = muk;
        for (int i = 0; i < count; i++) {
            lstar += alphak * exp(-betak * (t - kevents[i]));
        }

        // draw inter-arrival: U ~ Uniform(0,1)
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        double u1 = (double)(seed >> 11) / (double)(1ULL << 53);
        if (u1 < 1e-300) u1 = 1e-300;
        double dt = -log(u1) / lstar;
        t += dt;

        if (t >= T_max) break;

        // recompute true intensity
        double lam = muk;
        for (int i = 0; i < count; i++) {
            lam += alphak * exp(-betak * (t - kevents[i]));
        }

        // accept/reject
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        double u2 = (double)(seed >> 11) / (double)(1ULL << 53);
        if (u2 <= lam / lstar) {
            kevents[count++] = t;
        }
    }
}

// ---------------------------------------------------------------------------
// C wrappers
// ---------------------------------------------------------------------------

extern "C" {

int cuda_hawkes_intensity(
    const double* times, const double* alpha,
    const double* beta,  const double* mu,
    double t,
    double* out,
    int K, int T
) {
    double *dT, *dAlpha, *dBeta, *dMu, *dOut;
    if (alloc_copy(times, (void**)&dT,     (size_t)T*sizeof(double))) return -1;
    if (alloc_copy(alpha, (void**)&dAlpha, (size_t)K*sizeof(double))) { cudaFree(dT); return -1; }
    if (alloc_copy(beta,  (void**)&dBeta,  (size_t)K*sizeof(double))) { cudaFree(dT); cudaFree(dAlpha); return -1; }
    if (alloc_copy(mu,    (void**)&dMu,    (size_t)K*sizeof(double))) { cudaFree(dT); cudaFree(dAlpha); cudaFree(dBeta); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dOut, (size_t)K*sizeof(double)));
    int grid = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    intensity_kernel<<<grid, BLOCK_SIZE>>>(dT, dAlpha, dBeta, dMu, dOut, t, K, T);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, dOut, (size_t)K*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dT); cudaFree(dAlpha); cudaFree(dBeta); cudaFree(dMu); cudaFree(dOut);
    return 0;
}

int cuda_hawkes_kernel_matrix(
    const double* times,
    double alpha, double beta,
    double* out,
    int T
) {
    double *dTimes, *dOut;
    size_t nb = (size_t)T * sizeof(double);
    size_t mo = (size_t)T * T * sizeof(double);
    if (alloc_copy(times, (void**)&dTimes, nb)) return -1;
    CUDA_CHECK(cudaMalloc((void**)&dOut, mo));
    CUDA_CHECK(cudaMemset(dOut, 0, mo));
    int total = T * T;
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_matrix_kernel<<<grid, BLOCK_SIZE>>>(dTimes, alpha, beta, dOut, T);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, dOut, mo, cudaMemcpyDeviceToHost));
    cudaFree(dTimes); cudaFree(dOut);
    return 0;
}

int cuda_hawkes_log_likelihood(
    const double* intensities,
    double integral,
    double* out,
    int T
) {
    double *dIntens, *dSum;
    size_t nb = (size_t)T * sizeof(double);
    if (alloc_copy(intensities, (void**)&dIntens, nb)) return -1;

    CUDA_CHECK(cudaMalloc((void**)&dSum, sizeof(double)));
    CUDA_CHECK(cudaMemset(dSum, 0, sizeof(double)));

    int numBlocks = (T + BLOCK_SIZE - 1) / BLOCK_SIZE;
    log_sum_kernel<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(dIntens, dSum, T);
    CUDA_CHECK(cudaGetLastError());

    double sumLog = 0.0;
    CUDA_CHECK(cudaMemcpy(&sumLog, dSum, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dSum);

    out[0] = sumLog - integral;
    cudaFree(dIntens);
    return 0;
}

int cuda_hawkes_simulate(
    const double* mu, const double* alpha,
    const double* beta,
    double T_max, int K, int maxSteps,
    double* out
) {
    double *dMu, *dAlpha, *dBeta, *dOut;
    if (alloc_copy(mu,    (void**)&dMu,    (size_t)K*sizeof(double))) return -1;
    if (alloc_copy(alpha, (void**)&dAlpha, (size_t)K*sizeof(double))) { cudaFree(dMu); return -1; }
    if (alloc_copy(beta,  (void**)&dBeta,  (size_t)K*sizeof(double))) { cudaFree(dMu); cudaFree(dAlpha); return -1; }
    size_t osz = (size_t)K * maxSteps * sizeof(double);
    CUDA_CHECK(cudaMalloc((void**)&dOut, osz));

    int totalInit = K * maxSteps;
    int gridInit = (totalInit + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hawkes_fill_neg_one_kernel<<<gridInit, BLOCK_SIZE>>>(dOut, totalInit);
    CUDA_CHECK(cudaGetLastError());

    int grid = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    simulate_kernel<<<grid, BLOCK_SIZE>>>(dMu, dAlpha, dBeta, T_max, dOut, K, maxSteps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, dOut, osz, cudaMemcpyDeviceToHost));
    cudaFree(dMu); cudaFree(dAlpha); cudaFree(dBeta); cudaFree(dOut);
    return 0;
}

} // extern "C"
