#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>
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

__global__ void split_kernel(
    const double* src, double* dst,
    int outer, int dim_size, int split_size, int inner, int total)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;

    int element_in_chunk = split_size * inner;
    int chunk_elements = outer * element_in_chunk;

    int chunk = index / chunk_elements;
    int chunk_offset = index - chunk * chunk_elements;
    int outer_index = chunk_offset / element_in_chunk;
    int within = chunk_offset - outer_index * element_in_chunk;
    int src_index = (outer_index * dim_size + chunk * split_size) * inner + within;
    dst[index] = src[src_index];
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

static int checked_mul_size(size_t* out, size_t factor) {
    if (*out > SIZE_MAX / factor) return -1;

    *out *= factor;
    return 0;
}

static int checked_total4(size_t* total, int a, int b, int c, int d) {
    if (a <= 0 || b <= 0 || c <= 0 || d <= 0) return -1;

    *total = (size_t)a;
    if (checked_mul_size(total, (size_t)b)) return -1;
    if (checked_mul_size(total, (size_t)c)) return -1;
    if (checked_mul_size(total, (size_t)d)) return -1;
    if (*total > INT_MAX) return -1;

    return 0;
}

// ---------------------------------------------------------------------------
// C linkage wrappers
// ---------------------------------------------------------------------------

extern "C" {

int cuda_transpose(const double* src, double* dst,
                   const int* shape, int rank,
                   int dim0, int dim1, int n)
{
    if (!src || !dst || !shape || rank <= 0 || rank > MAX_RANK || n <= 0) return -1;
    if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank) return -1;

    size_t total_items = 1;

    for (int index = 0; index < rank; index++) {
        if (shape[index] <= 0) return -1;
        if (checked_mul_size(&total_items, (size_t)shape[index])) return -1;
        if (total_items > INT_MAX) return -1;
    }

    if (total_items != (size_t)n) return -1;

    double *d_src = NULL, *d_dst = NULL;
    int    *d_shapepad = NULL;
    size_t bytes       = total_items * sizeof(double);

    if (cudaMalloc(&d_src,   bytes)       != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst,   bytes)       != cudaSuccess) goto fail;
    if (cudaMalloc(&d_shapepad, MAX_RANK * sizeof(int)) != cudaSuccess) goto fail;

    if (cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_shapepad, shape, (size_t)rank * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    if (rank < MAX_RANK) {
        if (cudaMemset(d_shapepad + rank, 0,
                       (size_t)(MAX_RANK - rank) * sizeof(int)) != cudaSuccess) {
            goto fail;
        }
    }

    transpose_kernel<<<blocks(n), BLOCK>>>(d_src, d_dst, rank, dim0, dim1, d_shapepad, n);

    if (cudaGetLastError()     != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_shapepad);
    return 0;
fail:
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_shapepad);
    return -1;
}

int cuda_copy(const double* src, double* dst, int n) {
    if (!src || !dst || n <= 0) return -1;

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
    if (!dst || n_a < 0 || n_b < 0) return -1;
    if (n_a > 0 && !srcA) return -1;
    if (n_b > 0 && !srcB) return -1;

    size_t total_items = (size_t)n_a + (size_t)n_b;

    if (total_items == 0 || total_items > INT_MAX) return -1;

    double *d_a = NULL, *d_b = NULL, *d_dst = NULL;
    size_t a_bytes   = (size_t)n_a * sizeof(double);
    size_t b_bytes   = (size_t)n_b * sizeof(double);
    size_t dst_bytes = (size_t)(n_a + n_b) * sizeof(double);
    int total = (int)total_items;

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

int cuda_split(const double* src, double* dst,
               int outer, int dim_size, int split_size, int inner)
{
    if (!src || !dst || outer <= 0 || dim_size <= 0 || split_size <= 0 || inner <= 0) {
        return -1;
    }

    if (split_size > dim_size || dim_size % split_size != 0) return -1;

    size_t total_items = (size_t)outer * (size_t)dim_size * (size_t)inner;

    if (total_items > INT_MAX) return -1;

    int total = (int)total_items;
    double *d_src = NULL, *d_dst = NULL;
    size_t bytes = total_items * sizeof(double);

    if (cudaMalloc(&d_src, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, bytes) != cudaSuccess) { cudaFree(d_src); return -1; }

    if (cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    split_kernel<<<blocks(total), BLOCK>>>(
        d_src, d_dst, outer, dim_size, split_size, inner, total
    );
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src); cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src); cudaFree(d_dst);
    return -1;
}

int cuda_view_as_heads(const double* src, double* dst,
                       int B, int T, int H, int head_dim)
{
    if (!src || !dst) return -1;

    size_t total_items = 0;

    if (checked_total4(&total_items, B, T, H, head_dim)) return -1;

    int n = (int)total_items;
    double *d_src = NULL, *d_dst = NULL;
    size_t bytes = total_items * sizeof(double);

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
    if (!src || !dst) return -1;

    size_t total_items = 0;

    if (checked_total4(&total_items, B, H, T, head_dim)) return -1;

    int n = (int)total_items;
    double *d_src = NULL, *d_dst = NULL;
    size_t bytes = total_items * sizeof(double);

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
