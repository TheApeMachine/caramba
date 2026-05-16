#include <cuda_runtime.h>
#include <math.h>
#include "positional.h"

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// RoPE: each thread handles one rotation pair.
// idx = (b_h * seq_len + t) * num_pairs + i
__global__ void rope_kernel(
    const double* x,
    double*       out,
    const double* cos_table,
    const double* sin_table,
    int           seq_len,
    int           head_dim,
    int           rope_mode,
    int           total_threads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_threads) return;

    int num_pairs = head_dim / 2;
    int pair_i    = idx % num_pairs;             // dimension pair index
    int slot      = idx / num_pairs;             // b_h * seq_len + t
    int t         = slot % seq_len;

    int slot_base = slot * head_dim;
    int first_idx = slot_base + pair_i * 2;
    int second_idx = first_idx + 1;

    if (rope_mode == 1) {
        first_idx = slot_base + pair_i;
        second_idx = first_idx + num_pairs;
    }

    double x0 = x[first_idx];
    double x1 = x[second_idx];

    int tbl_idx = t * num_pairs + pair_i;
    double c    = cos_table[tbl_idx];
    double s    = sin_table[tbl_idx];

    out[first_idx]  = x0 * c - x1 * s;
    out[second_idx] = x0 * s + x1 * c;
}

// ALiBi: each thread computes one output element.
// idx = h * seq_len_q * seq_len_k + q * seq_len_k + k
__global__ void alibi_kernel(
    double*       out,
    const double* slopes,
    int           seq_len_q,
    int           seq_len_k,
    int           num_heads,
    int           causal,
    int           total_threads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_threads) return;

    int total_qk = seq_len_q * seq_len_k;
    int h = idx / total_qk;
    int rem = idx % total_qk;
    int q = rem / seq_len_k;
    int k = rem % seq_len_k;

    double val = slopes[h] * (double)(k - q);
    if (!causal && val < 0.0) val = -val;
    out[idx] = val;
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

static const int BLOCK = 256;
static inline int blocks(int n) { return (n + BLOCK - 1) / BLOCK; }

// ---------------------------------------------------------------------------
// C linkage wrappers
// ---------------------------------------------------------------------------

extern "C" {

int cuda_rope(
    const double* x,
    double*       out,
    const double* cos_table,
    const double* sin_table,
    int           seq_len,
    int           head_dim,
    int           rope_mode,
    int           total_heads)
{
    int num_pairs    = head_dim / 2;
    int total_n      = total_heads * seq_len * head_dim;
    int grid_threads = total_heads * seq_len * num_pairs;
    int tbl_n        = seq_len * num_pairs;

    size_t xbytes   = (size_t)total_n * sizeof(double);
    size_t tblbytes = (size_t)tbl_n * sizeof(double);

    double *d_x = NULL, *d_out = NULL, *d_cos = NULL, *d_sin = NULL;
    if (cudaMalloc(&d_x,   xbytes)   != cudaSuccess) return -1;
    if (cudaMalloc(&d_out, xbytes)   != cudaSuccess) goto fail;
    if (cudaMalloc(&d_cos, tblbytes) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_sin, tblbytes) != cudaSuccess) goto fail;

    if (cudaMemcpy(d_x,   x,         xbytes,   cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_cos, cos_table, tblbytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_sin, sin_table, tblbytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    rope_kernel<<<blocks(grid_threads), BLOCK>>>(
        d_x, d_out, d_cos, d_sin, seq_len, head_dim, rope_mode, grid_threads);
    if (cudaGetLastError()    != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;

    if (cudaMemcpy(out, d_out, xbytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_x); cudaFree(d_out); cudaFree(d_cos); cudaFree(d_sin);
    return 0;
fail:
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_cos); cudaFree(d_sin);
    return -1;
}

int cuda_alibi(
    double*       out,
    const double* slopes,
    int           num_heads,
    int           seq_len_q,
    int           seq_len_k,
    int           causal)
{
    int total = num_heads * seq_len_q * seq_len_k;
    size_t outbytes    = (size_t)total * sizeof(double);
    size_t slopebytes  = (size_t)num_heads * sizeof(double);

    double *d_out = NULL, *d_slopes = NULL;
    if (cudaMalloc(&d_out,    outbytes)   != cudaSuccess) return -1;
    if (cudaMalloc(&d_slopes, slopebytes) != cudaSuccess) goto fail;

    if (cudaMemcpy(d_slopes, slopes, slopebytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    alibi_kernel<<<blocks(total), BLOCK>>>(
        d_out, d_slopes, seq_len_q, seq_len_k, num_heads, causal, total);
    if (cudaGetLastError()    != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;

    if (cudaMemcpy(out, d_out, outbytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_out); cudaFree(d_slopes);
    return 0;
fail:
    cudaFree(d_out); cudaFree(d_slopes);
    return -1;
}

} // extern "C"
