#include <cuda_runtime.h>
#include "embedding.h"

// ---------------------------------------------------------------------------
// Device kernel
// Each thread handles one (token, dim) pair.
//   idx = token_index * d_model + dim_index
// ---------------------------------------------------------------------------

__global__ void token_embedding_kernel(
    const double* weight,
    const double* tokens,
    double*       out,
    int           d_model,
    int           n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * d_model;
    if (idx >= total) return;

    int token_idx = idx / d_model;
    int dim_idx   = idx % d_model;
    int token_id  = (int)tokens[token_idx];
    out[idx] = weight[token_id * d_model + dim_idx];
}

// ---------------------------------------------------------------------------
// C linkage wrapper
// ---------------------------------------------------------------------------

static const int BLOCK = 256;

extern "C" {

int cuda_token_embedding(
    const double* tokens,
    double*       out,
    const double* weight,
    int           n,
    int           d_model,
    int           vocab_size)
{
    int total = n * d_model;
    size_t tok_bytes    = (size_t)n * sizeof(double);
    size_t out_bytes    = (size_t)total * sizeof(double);
    size_t weight_bytes = (size_t)(vocab_size * d_model) * sizeof(double);

    double *d_tokens = NULL, *d_out = NULL, *d_weight = NULL;

    if (cudaMalloc(&d_tokens, tok_bytes)    != cudaSuccess) return -1;
    if (cudaMalloc(&d_out,    out_bytes)    != cudaSuccess) { cudaFree(d_tokens); return -1; }
    if (cudaMalloc(&d_weight, weight_bytes) != cudaSuccess) {
        cudaFree(d_tokens); cudaFree(d_out); return -1;
    }

    if (cudaMemcpy(d_tokens, tokens, tok_bytes,    cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_weight, weight, weight_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    {
        int grid = (total + BLOCK - 1) / BLOCK;
        token_embedding_kernel<<<grid, BLOCK>>>(d_weight, d_tokens, d_out, d_model, n);
    }

    if (cudaGetLastError()       != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize()  != cudaSuccess) goto fail;

    if (cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_tokens);
    cudaFree(d_out);
    cudaFree(d_weight);
    return 0;

fail:
    cudaFree(d_tokens);
    cudaFree(d_out);
    cudaFree(d_weight);
    return -1;
}

} // extern "C"
