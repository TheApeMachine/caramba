#include <cuda_runtime.h>
#include <math.h>
#include "masking.h"

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

__global__ void causal_mask_kernel(double* out, int seq_len) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len || col >= seq_len) return;
    out[row * seq_len + col] = (col <= row) ? 0.0 : -__builtin_huge_val();
}

__global__ void apply_mask_kernel(const double* scores, const double* mask, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = scores[i] + mask[i];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const int BLOCK = 256;
static const int BLOCK2D = 16;

static inline int blocks(int n) { return (n + BLOCK - 1) / BLOCK; }

// ---------------------------------------------------------------------------
// C linkage wrappers
// ---------------------------------------------------------------------------

extern "C" {

int cuda_causal_mask(double* out, int seq_len) {
    size_t bytes = (size_t)seq_len * (size_t)seq_len * sizeof(double);
    double* d_out = NULL;

    if (cudaMalloc(&d_out, bytes) != cudaSuccess) return -1;

    dim3 block(BLOCK2D, BLOCK2D);
    dim3 grid((seq_len + BLOCK2D - 1) / BLOCK2D,
              (seq_len + BLOCK2D - 1) / BLOCK2D);
    causal_mask_kernel<<<grid, block>>>(d_out, seq_len);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_out);
    return 0;
fail:
    cudaFree(d_out);
    return -1;
}

int cuda_apply_mask(const double* scores, const double* mask, double* out, int n) {
    size_t bytes = (size_t)n * sizeof(double);
    double *d_scores = NULL, *d_mask = NULL, *d_out = NULL;

    if (cudaMalloc(&d_scores, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_mask,   bytes) != cudaSuccess) { cudaFree(d_scores); return -1; }
    if (cudaMalloc(&d_out,    bytes) != cudaSuccess) { cudaFree(d_scores); cudaFree(d_mask); return -1; }

    if (cudaMemcpy(d_scores, scores, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_mask,   mask,   bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    apply_mask_kernel<<<blocks(n), BLOCK>>>(d_scores, d_mask, d_out, n);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_scores);
    cudaFree(d_mask);
    cudaFree(d_out);
    return 0;
fail:
    cudaFree(d_scores);
    cudaFree(d_mask);
    cudaFree(d_out);
    return -1;
}

} // extern "C"
