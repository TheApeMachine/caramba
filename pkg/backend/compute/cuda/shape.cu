#include <cuda_runtime.h>
#include "shape.h"

// Max rank supported by the general transpose kernel.
#define MAX_RANK 8

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// General N-D transpose: each thread handles one input element and writes it
// to the correct output location after swapping dim0 and dim1.
__global__ void transpose_kernel(
    const double* src, double* dst,
    int rank, int dim0, int dim1,
    const int* shape, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Compute input strides (row-major).
    int strides[MAX_RANK];
    strides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; d--)
        strides[d] = strides[d + 1] * shape[d + 1];

    // Decode flat index into multi-dimensional coords.
    int coords[MAX_RANK];
    int rem = idx;
    for (int d = 0; d < rank; d++) {
        coords[d] = rem / strides[d];
        rem       = rem % strides[d];
    }

    // Swap dim0 and dim1.
    int tmp      = coords[dim0];
    coords[dim0] = coords[dim1];
    coords[dim1] = tmp;

    // Compute output shape and strides.
    int outShape[MAX_RANK];
    for (int d = 0; d < rank; d++) outShape[d] = shape[d];
    outShape[dim0] = shape[dim1];
    outShape[dim1] = shape[dim0];

    int outStrides[MAX_RANK];
    outStrides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; d--)
        outStrides[d] = outStrides[d + 1] * outShape[d + 1];

    int outIdx = 0;
    for (int d = 0; d < rank; d++)
        outIdx += coords[d] * outStrides[d];

    dst[outIdx] = src[idx];
}

__global__ void copy_kernel(const double* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void concat_kernel(
    const double* srcA, int n_a,
    const double* srcB, int n_b,
    double* dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_a + n_b;
    if (i >= total) return;
    dst[i] = (i < n_a) ? srcA[i] : srcB[i - n_a];
}

// ViewAsHeads: input [B,T,H,head_dim] -> output [B,H,T,head_dim].
__global__ void view_as_heads_kernel(
    const double* src, double* dst,
    int B, int T, int H, int head_dim, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int hd  = idx % head_dim;
    int rem = idx / head_dim;
    int h   = rem % H;
    rem     = rem / H;
    int t   = rem % T;
    int b   = rem / T;

    int outIdx = b * (H * T * head_dim)
               + h * (T * head_dim)
               + t * head_dim
               + hd;

    dst[outIdx] = src[idx];
}

// MergeHeads: input [B,H,T,head_dim] -> output [B,T,H,head_dim].
__global__ void merge_heads_kernel(
    const double* src, double* dst,
    int B, int H, int T, int head_dim, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int hd  = idx % head_dim;
    int rem = idx / head_dim;
    int t   = rem % T;
    rem     = rem / T;
    int h   = rem % H;
    int b   = rem / H;

    int outIdx = b * (T * H * head_dim)
               + t * (H * head_dim)
               + h * head_dim
               + hd;

    dst[outIdx] = src[idx];
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

int cuda_transpose(const double* src, double* dst,
                   const int* shape, int rank,
                   int dim0, int dim1, int n)
{
    double *d_src = NULL, *d_dst = NULL;
    int    *d_shape = NULL;
    size_t bytes       = (size_t)n * sizeof(double);
    size_t shape_bytes = (size_t)rank * sizeof(int);

    if (cudaMalloc(&d_src,   bytes)       != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst,   bytes)       != cudaSuccess) goto fail;
    if (cudaMalloc(&d_shape, shape_bytes) != cudaSuccess) goto fail;

    if (cudaMemcpy(d_src,   src,   bytes,       cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_shape, shape, shape_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    // Pass shape as __shared__ via a fixed-size array.  For device-side, we
    // pass it as a regular device pointer and the kernel reads it.
    // We need to adapt the kernel signature: use a pointer in constant memory.
    {
        // Use a temporary host-side array padded to MAX_RANK.
        int h_shape[MAX_RANK] = {};
        for (int i = 0; i < rank && i < MAX_RANK; i++) h_shape[i] = shape[i];

        int *d_shapepad = NULL;
        if (cudaMalloc(&d_shapepad, MAX_RANK * sizeof(int)) != cudaSuccess) goto fail;
        if (cudaMemcpy(d_shapepad, h_shape, MAX_RANK * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_shapepad);
            goto fail;
        }
        transpose_kernel<<<blocks(n), BLOCK>>>(d_src, d_dst, rank, dim0, dim1, d_shapepad, n);
        cudaFree(d_shapepad);
    }

    if (cudaGetLastError()     != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_shape);
    return 0;
fail:
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_shape);
    return -1;
}

int cuda_copy(const double* src, double* dst, int n) {
    double *d_src = NULL, *d_dst = NULL;
    size_t bytes = (size_t)n * sizeof(double);

    if (cudaMalloc(&d_src, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, bytes) != cudaSuccess) { cudaFree(d_src); return -1; }

    if (cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    copy_kernel<<<blocks(n), BLOCK>>>(d_src, d_dst, n);
    if (cudaGetLastError()     != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src); cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src); cudaFree(d_dst);
    return -1;
}

int cuda_concat(const double* srcA, int n_a,
                const double* srcB, int n_b,
                double* dst)
{
    double *d_a = NULL, *d_b = NULL, *d_dst = NULL;
    size_t a_bytes   = (size_t)n_a * sizeof(double);
    size_t b_bytes   = (size_t)n_b * sizeof(double);
    size_t dst_bytes = (size_t)(n_a + n_b) * sizeof(double);
    int total = n_a + n_b;

    if (cudaMalloc(&d_a,   a_bytes)   != cudaSuccess) return -1;
    if (cudaMalloc(&d_b,   b_bytes)   != cudaSuccess) goto fail;
    if (cudaMalloc(&d_dst, dst_bytes) != cudaSuccess) goto fail;

    if (cudaMemcpy(d_a, srcA, a_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_b, srcB, b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    concat_kernel<<<blocks(total), BLOCK>>>(d_a, n_a, d_b, n_b, d_dst);
    if (cudaGetLastError()     != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_dst);
    return -1;
}

int cuda_view_as_heads(const double* src, double* dst,
                       int B, int T, int H, int head_dim)
{
    int n = B * T * H * head_dim;
    double *d_src = NULL, *d_dst = NULL;
    size_t bytes = (size_t)n * sizeof(double);

    if (cudaMalloc(&d_src, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, bytes) != cudaSuccess) { cudaFree(d_src); return -1; }

    if (cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    view_as_heads_kernel<<<blocks(n), BLOCK>>>(d_src, d_dst, B, T, H, head_dim, n);
    if (cudaGetLastError()     != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src); cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src); cudaFree(d_dst);
    return -1;
}

int cuda_merge_heads(const double* src, double* dst,
                     int B, int H, int T, int head_dim)
{
    int n = B * H * T * head_dim;
    double *d_src = NULL, *d_dst = NULL;
    size_t bytes = (size_t)n * sizeof(double);

    if (cudaMalloc(&d_src, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, bytes) != cudaSuccess) { cudaFree(d_src); return -1; }

    if (cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    merge_heads_kernel<<<blocks(n), BLOCK>>>(d_src, d_dst, B, H, T, head_dim, n);
    if (cudaGetLastError()     != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src); cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src); cudaFree(d_dst);
    return -1;
}

} // extern "C"
