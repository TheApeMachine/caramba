#include <cuda_runtime.h>
#include "projection.h"

// ---------------------------------------------------------------------------
// Tiled matmul kernel: C[M,N] = A[M,K] @ B[K,N] (row-major, double)
// Each thread computes one C[row,col].
// TILE=16 threadblock.
// ---------------------------------------------------------------------------

#define TILE 16

__global__ void proj_matmul_kernel(
    const double* __restrict__ A,   // [M*K]
    const double* __restrict__ B,   // [K*N]
    double* __restrict__       C,   // [M*N]
    int M, int K, int N)
{
    __shared__ double sA[TILE][TILE];
    __shared__ double sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    double acc = 0.0;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();
        for (int k = 0; k < TILE; k++) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Linear/FusedQKV kernel: C = A @ W^T + bias
// W is stored [N*K], so W^T[K,N] access is W[col*K + k].
// ---------------------------------------------------------------------------

__global__ void proj_linear_kernel(
    const double* __restrict__ A,   // [M*K]
    const double* __restrict__ W,   // [N*K]  (W^T implied)
    const double* __restrict__ bias,// [N] or NULL
    double* __restrict__       C,   // [M*N]
    int M, int K, int N, int has_bias)
{
    __shared__ double sA[TILE][TILE];
    __shared__ double sW[TILE][TILE];  // sW[t_k][t_n] = W[col*K + k]

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    double acc = 0.0;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int wCol = t * TILE + threadIdx.y;  // k index for W[col*K + k]
        // A[row, aCol]
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0;
        // W^T[wCol, col] = W[col * K + wCol]
        int w_col_idx = blockIdx.x * TILE + threadIdx.x;  // column in output (= vocab/out idx)
        sW[threadIdx.y][threadIdx.x] = (wCol < K && w_col_idx < N) ? W[w_col_idx * K + wCol] : 0.0;
        __syncthreads();
        for (int k = 0; k < TILE; k++) {
            acc += sA[threadIdx.y][k] * sW[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        if (has_bias) acc += bias[col];
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

static const int BLOCK = TILE;

static int launch_linear(
    const double* h_src,    int src_n,
    const double* h_weight, int weight_n,
    const double* h_bias,   int bias_n,
    double* h_dst,          int dst_n,
    int M, int K, int N, int has_bias)
{
    double *d_src = NULL, *d_weight = NULL, *d_bias = NULL, *d_dst = NULL;
    size_t src_bytes    = (size_t)src_n    * sizeof(double);
    size_t weight_bytes = (size_t)weight_n * sizeof(double);
    size_t dst_bytes    = (size_t)dst_n    * sizeof(double);

    if (cudaMalloc(&d_src,    src_bytes)    != cudaSuccess) goto fail;
    if (cudaMalloc(&d_weight, weight_bytes) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_dst,    dst_bytes)    != cudaSuccess) goto fail;

    if (cudaMemcpy(d_src,    h_src,    src_bytes,    cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_weight, h_weight, weight_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    if (has_bias && h_bias) {
        size_t bias_bytes = (size_t)bias_n * sizeof(double);
        if (cudaMalloc(&d_bias, bias_bytes) != cudaSuccess) goto fail;
        if (cudaMemcpy(d_bias, h_bias, bias_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    }

    {
        dim3 block(BLOCK, BLOCK);
        dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
        proj_linear_kernel<<<grid, block>>>(d_src, d_weight, d_bias, d_dst, M, K, N, has_bias);
    }
    if (cudaGetLastError()       != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize()  != cudaSuccess) goto fail;
    if (cudaMemcpy(h_dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src); cudaFree(d_weight); cudaFree(d_dst); if (d_bias) cudaFree(d_bias);
    return 0;
fail:
    if (d_src)    cudaFree(d_src);
    if (d_weight) cudaFree(d_weight);
    if (d_dst)    cudaFree(d_dst);
    if (d_bias)   cudaFree(d_bias);
    return -1;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

int cuda_linear(const double* src, const double* weight, const double* bias,
                double* dst, int M, int K, int N, int has_bias)
{
    return launch_linear(src, M*K, weight, N*K, bias, N, dst, M*N, M, K, N, has_bias);
}

int cuda_fused_qkv(const double* src, const double* weight, const double* bias,
                   double* dst, int M, int K, int N, int has_bias)
{
    return launch_linear(src, M*K, weight, N*K, bias, N, dst, M*N, M, K, N, has_bias);
}

int cuda_tied_embedding(const double* src, const double* weight,
                        double* dst, int M, int D, int V)
{
    return launch_linear(src, M*D, weight, V*D, NULL, 0, dst, M*V, M, D, V, 0);
}

} // extern "C"
