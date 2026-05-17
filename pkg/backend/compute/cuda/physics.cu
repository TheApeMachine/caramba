#include <cuda_runtime.h>
#include "physics.h"

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) return -1; \
} while(0)

static int alloc_copy(const void* h_src, void** d_ptr, size_t bytes) {
    if (cudaMalloc(d_ptr, bytes) != cudaSuccess) return -1;
    if (cudaMemcpy(*d_ptr, h_src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*d_ptr);
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// 1D periodic Laplacian
//   dst[i] = (src[(i-1+n)%n] + src[(i+1)%n] - 2*src[i]) * inv_h2
// ---------------------------------------------------------------------------
__global__ void laplacian_1d_kernel(const double* __restrict__ src,
                                    double* __restrict__ dst,
                                    int n, double inv_h2)
{
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;

    int left  = (i == 0)     ? n - 1 : i - 1;
    int right = (i == n - 1) ? 0     : i + 1;

    dst[i] = (src[left] + src[right] - 2.0 * src[i]) * inv_h2;
}

// ---------------------------------------------------------------------------
// 2D periodic 5-point Laplacian on row-major [H, W]
// ---------------------------------------------------------------------------
__global__ void laplacian_2d_kernel(const double* __restrict__ src,
                                    double* __restrict__ dst,
                                    int H, int W, double inv_h2)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = H * W;
    if (idx >= total) return;

    int i = idx / W;
    int j = idx - i * W;

    int up    = (i == 0)     ? H - 1 : i - 1;
    int down  = (i == H - 1) ? 0     : i + 1;
    int left  = (j == 0)     ? W - 1 : j - 1;
    int right = (j == W - 1) ? 0     : j + 1;

    double center = src[idx];
    double horizontal = src[i * W + left] + src[i * W + right];
    double vertical   = src[up * W + j]   + src[down * W + j];

    dst[idx] = (horizontal + vertical - 4.0 * center) * inv_h2;
}

// ---------------------------------------------------------------------------
// 3D periodic 7-point Laplacian on row-major [D, H, W]
// ---------------------------------------------------------------------------
__global__ void laplacian_3d_kernel(const double* __restrict__ src,
                                    double* __restrict__ dst,
                                    int D, int H, int W, double inv_h2)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int hw = H * W;
    int total = D * hw;
    if (idx >= total) return;

    int k = idx / hw;
    int rem = idx - k * hw;
    int i = rem / W;
    int j = rem - i * W;

    int front = (k == 0)     ? D - 1 : k - 1;
    int back  = (k == D - 1) ? 0     : k + 1;
    int up    = (i == 0)     ? H - 1 : i - 1;
    int down  = (i == H - 1) ? 0     : i + 1;
    int left  = (j == 0)     ? W - 1 : j - 1;
    int right = (j == W - 1) ? 0     : j + 1;

    double center = src[idx];
    double horizontal = src[k * hw + i * W + left] + src[k * hw + i * W + right];
    double vertical   = src[k * hw + up * W + j]   + src[k * hw + down * W + j];
    double transverse = src[front * hw + i * W + j] + src[back * hw + i * W + j];

    dst[idx] = (horizontal + vertical + transverse - 6.0 * center) * inv_h2;
}

// ---------------------------------------------------------------------------
// C wrappers — copy host→device, launch, copy device→host
// ---------------------------------------------------------------------------
extern "C" {

int cuda_laplacian_1d(const double* src, double* dst, int n, double inv_h2) {
    if (n <= 0) return -1;
    if (!src || !dst) return -1;

    double *d_src = nullptr, *d_dst = nullptr;
    size_t bytes = (size_t)n * sizeof(double);

    if (alloc_copy(src, (void**)&d_src, bytes)) return -1;
    if (cudaMalloc((void**)&d_dst, bytes) != cudaSuccess) {
        cudaFree(d_src);
        return -1;
    }

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    laplacian_1d_kernel<<<blocks, BLOCK_SIZE>>>(d_src, d_dst, n, inv_h2);

    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_src); cudaFree(d_dst);
        return -1;
    }
    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_src); cudaFree(d_dst);
        return -1;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}

int cuda_laplacian_2d(const double* src, double* dst, int H, int W, double inv_h2) {
    if (H <= 0 || W <= 0) return -1;
    if (!src || !dst) return -1;

    int total = H * W;
    double *d_src = nullptr, *d_dst = nullptr;
    size_t bytes = (size_t)total * sizeof(double);

    if (alloc_copy(src, (void**)&d_src, bytes)) return -1;
    if (cudaMalloc((void**)&d_dst, bytes) != cudaSuccess) {
        cudaFree(d_src);
        return -1;
    }

    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    laplacian_2d_kernel<<<blocks, BLOCK_SIZE>>>(d_src, d_dst, H, W, inv_h2);

    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_src); cudaFree(d_dst);
        return -1;
    }
    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_src); cudaFree(d_dst);
        return -1;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}

int cuda_laplacian_3d(const double* src, double* dst, int D, int H, int W, double inv_h2) {
    if (D <= 0 || H <= 0 || W <= 0) return -1;
    if (!src || !dst) return -1;

    int total = D * H * W;
    double *d_src = nullptr, *d_dst = nullptr;
    size_t bytes = (size_t)total * sizeof(double);

    if (alloc_copy(src, (void**)&d_src, bytes)) return -1;
    if (cudaMalloc((void**)&d_dst, bytes) != cudaSuccess) {
        cudaFree(d_src);
        return -1;
    }

    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    laplacian_3d_kernel<<<blocks, BLOCK_SIZE>>>(d_src, d_dst, D, H, W, inv_h2);

    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_src); cudaFree(d_dst);
        return -1;
    }
    if (cudaMemcpy(dst, d_dst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_src); cudaFree(d_dst);
        return -1;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}

} // extern "C"
