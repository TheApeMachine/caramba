#include <cuda_runtime.h>
#include <math.h>
#include "math.h"

#define TILE_SIZE 16
#define BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// Helper macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) return -1; \
} while(0)

// ---------------------------------------------------------------------------
// matmul_kernel: tiled [M,K]x[K,N]->[M,N]
// ---------------------------------------------------------------------------
__global__ void matmul_kernel(const double* __restrict__ A,
                               const double* __restrict__ B,
                               double* __restrict__ C,
                               int M, int K, int N)
{
    __shared__ double tA[TILE_SIZE][TILE_SIZE];
    __shared__ double tB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    double acc = 0.0;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;
        tA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0;
        tB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++) acc += tA[threadIdx.y][i] * tB[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

__device__ double gelu_device(double x) {
    double x3 = x * x * x;
    double z = 0.7978845608028654 * (x + 0.044715 * x3);
    return 0.5 * x * (1.0 + tanh(z));
}

__global__ void matmul_add_kernel(const double* __restrict__ A,
                                  const double* __restrict__ B,
                                  const double* __restrict__ bias,
                                  double* __restrict__ C,
                                  int M, int K, int N,
                                  int bias_n, int apply_gelu)
{
    __shared__ double tA[TILE_SIZE][TILE_SIZE];
    __shared__ double tB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    double acc = 0.0;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;
        tA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0;
        tB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++) acc += tA[threadIdx.y][i] * tB[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        if (bias_n == N) {
            acc += bias[col];
        } else {
            acc += bias[row * N + col];
        }

        C[row * N + col] = apply_gelu ? gelu_device(acc) : acc;
    }
}

// ---------------------------------------------------------------------------
// Elementwise kernels
// ---------------------------------------------------------------------------
__global__ void add_kernel(const double* a, const double* b, double* out, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void mul_kernel(const double* a, const double* b, double* out, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void inv_sqrt_dim_scale_kernel(const double* src, double* dst, int n, double scale) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) dst[i] = src[i] * scale;
}

__global__ void exp_kernel(const double* src, double* dst, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) dst[i] = exp(src[i]);
}

__global__ void log_kernel(const double* src, double* dst, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) dst[i] = log(src[i]);
}

// ---------------------------------------------------------------------------
// softmax_kernel: one block per row, warp shuffle reductions
// ---------------------------------------------------------------------------
__global__ void softmax_kernel(const double* src, double* dst,
                                int dim_size)
{
    int row = blockIdx.x;
    const double* row_src = src + row * dim_size;
    double*       row_dst = dst + row * dim_size;

    // reduce max
    double lmax = -1e300;
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        double v = row_src[i];
        lmax = lmax > v ? lmax : v;
    }
    // warp reduce max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        double other = __shfl_down_sync(0xffffffff, lmax, offset);
        lmax = lmax > other ? lmax : other;
    }
    // block reduce via shared mem
    extern __shared__ double smem[];
    if (threadIdx.x % warpSize == 0) smem[threadIdx.x / warpSize] = lmax;
    __syncthreads();
    if (threadIdx.x < blockDim.x / warpSize) lmax = smem[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        double other = __shfl_down_sync(0xffffffff, lmax, offset);
        lmax = lmax > other ? lmax : other;
    }
    if (threadIdx.x == 0) smem[0] = lmax;
    __syncthreads();
    lmax = smem[0];
    __syncthreads();

    // exp and partial sum
    double lsum = 0.0;
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        double e = exp(row_src[i] - lmax);
        row_dst[i] = e;
        lsum += e;
    }
    // warp reduce sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        lsum += __shfl_down_sync(0xffffffff, lsum, offset);
    if (threadIdx.x % warpSize == 0) smem[threadIdx.x / warpSize] = lsum;
    __syncthreads();
    if (threadIdx.x < blockDim.x / warpSize) lsum = smem[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        lsum += __shfl_down_sync(0xffffffff, lsum, offset);
    if (threadIdx.x == 0) smem[0] = lsum;
    __syncthreads();
    lsum = smem[0];
    __syncthreads();

    for (int i = threadIdx.x; i < dim_size; i += blockDim.x)
        row_dst[i] /= lsum;
}

// ---------------------------------------------------------------------------
// layernorm_kernel: one block per row
// ---------------------------------------------------------------------------
__global__ void layernorm_kernel(const double* src, double* dst,
                                  const double* weight, const double* bias,
                                  int d_model, double eps)
{
    extern __shared__ double smem[];
    int row = blockIdx.x;
    const double* row_src = src + row * d_model;
    double*       row_dst = dst + row * d_model;

    // mean
    double lsum = 0.0;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) lsum += row_src[i];
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
        lsum += __shfl_down_sync(0xffffffff, lsum, offset);
    if (threadIdx.x % warpSize == 0) smem[threadIdx.x / warpSize] = lsum;
    __syncthreads();
    if (threadIdx.x < blockDim.x / warpSize) lsum = smem[threadIdx.x];
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
        lsum += __shfl_down_sync(0xffffffff, lsum, offset);
    if (threadIdx.x == 0) smem[0] = lsum;
    __syncthreads();
    double mean = smem[0] / d_model;
    __syncthreads();

    // variance
    double lvar = 0.0;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        double diff = row_src[i] - mean;
        lvar += diff * diff;
    }
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
        lvar += __shfl_down_sync(0xffffffff, lvar, offset);
    if (threadIdx.x % warpSize == 0) smem[threadIdx.x / warpSize] = lvar;
    __syncthreads();
    if (threadIdx.x < blockDim.x / warpSize) lvar = smem[threadIdx.x];
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
        lvar += __shfl_down_sync(0xffffffff, lvar, offset);
    if (threadIdx.x == 0) smem[0] = lvar;
    __syncthreads();
    double inv_std = rsqrt(smem[0] / d_model + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        row_dst[i] = (row_src[i] - mean) * inv_std * weight[i] + bias[i];
}

// ---------------------------------------------------------------------------
// rmsnorm_kernel: one block per row
// ---------------------------------------------------------------------------
__global__ void rmsnorm_kernel(const double* src, double* dst,
                                const double* weight,
                                int d_model, double eps)
{
    extern __shared__ double smem[];
    int row = blockIdx.x;
    const double* row_src = src + row * d_model;
    double*       row_dst = dst + row * d_model;

    double lss = 0.0;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        double v = row_src[i];
        lss += v * v;
    }
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
        lss += __shfl_down_sync(0xffffffff, lss, offset);
    if (threadIdx.x % warpSize == 0) smem[threadIdx.x / warpSize] = lss;
    __syncthreads();
    if (threadIdx.x < blockDim.x / warpSize) lss = smem[threadIdx.x];
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
        lss += __shfl_down_sync(0xffffffff, lss, offset);
    if (threadIdx.x == 0) smem[0] = lss;
    __syncthreads();
    double inv_rms = rsqrt(smem[0] / d_model + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
        row_dst[i] = row_src[i] * inv_rms * weight[i];
}

// ---------------------------------------------------------------------------
// C wrappers
// ---------------------------------------------------------------------------

static int alloc_copy_free(const void* h_src, void** d_ptr, size_t bytes) {
    if (cudaMalloc(d_ptr, bytes) != cudaSuccess) return -1;
    if (cudaMemcpy(*d_ptr, h_src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*d_ptr); return -1;
    }
    return 0;
}

extern "C" {

int cuda_matmul(const double* A, const double* B, double* C, int M, int K, int N) {
    double *dA, *dB, *dC;
    size_t ab = (size_t)M*K*sizeof(double);
    size_t bb = (size_t)K*N*sizeof(double);
    size_t cb = (size_t)M*N*sizeof(double);
    if (alloc_copy_free(A, (void**)&dA, ab)) return -1;
    if (alloc_copy_free(B, (void**)&dB, bb)) { cudaFree(dA); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dC, cb));
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
    matmul_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(C, dC, cb, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

int cuda_add(const double* a, const double* b, double* out, int n) {
    double *dA, *dB, *dOut;
    size_t nb = (size_t)n*sizeof(double);
    if (alloc_copy_free(a, (void**)&dA, nb)) return -1;
    if (alloc_copy_free(b, (void**)&dB, nb)) { cudaFree(dA); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dOut, nb));
    add_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dA, dB, dOut, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, dOut, nb, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dOut);
    return 0;
}

int cuda_mul(const double* a, const double* b, double* out, int n) {
    double *dA, *dB, *dOut;
    size_t nb = (size_t)n*sizeof(double);
    if (alloc_copy_free(a, (void**)&dA, nb)) return -1;
    if (alloc_copy_free(b, (void**)&dB, nb)) { cudaFree(dA); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dOut, nb));
    mul_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dA, dB, dOut, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, dOut, nb, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dOut);
    return 0;
}

int cuda_matmul_device(const void* A, const void* B, void* C, int M, int K, int N) {
    if (M == 0 || K == 0 || N == 0) return 0;
    if (!A || !B || !C) return -1;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
    matmul_kernel<<<grid, block>>>((const double*)A, (const double*)B, (double*)C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

int cuda_add_device(const void* a, const void* b, void* out, int n) {
    if (n == 0) return 0;
    if (!a || !b || !out) return -1;

    add_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        (const double*)a, (const double*)b, (double*)out, n
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

int cuda_mul_device(const void* a, const void* b, void* out, int n) {
    if (n == 0) return 0;
    if (!a || !b || !out) return -1;

    mul_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        (const double*)a, (const double*)b, (double*)out, n
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

int cuda_matmul_add_device(
    const void* A, const void* B, const void* bias, void* C,
    int M, int K, int N, int bias_n, int gelu
) {
    if (M == 0 || K == 0 || N == 0) return 0;
    if (!A || !B || !bias || !C) return -1;
    if (bias_n != N && bias_n != M*N) return -1;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
    matmul_add_kernel<<<grid, block>>>(
        (const double*)A, (const double*)B, (const double*)bias, (double*)C,
        M, K, N, bias_n, gelu
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

int cuda_inv_sqrt_dim_scale(const double* src, double* dst, int n, int dim) {
    double *dSrc, *dDst;
    size_t nb = (size_t)n*sizeof(double);
    double scale = 1.0 / sqrt((double)dim);
    if (alloc_copy_free(src, (void**)&dSrc, nb)) return -1;
    CUDA_CHECK(cudaMalloc((void**)&dDst, nb));
    inv_sqrt_dim_scale_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dSrc, dDst, n, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, nb, cudaMemcpyDeviceToHost));
    cudaFree(dSrc); cudaFree(dDst);
    return 0;
}

int cuda_exp(const double* src, double* dst, int n) {
    double *dSrc, *dDst;
    size_t nb = (size_t)n*sizeof(double);
    if (alloc_copy_free(src, (void**)&dSrc, nb)) return -1;
    CUDA_CHECK(cudaMalloc((void**)&dDst, nb));
    exp_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dSrc, dDst, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, nb, cudaMemcpyDeviceToHost));
    cudaFree(dSrc); cudaFree(dDst);
    return 0;
}

int cuda_log(const double* src, double* dst, int n) {
    double *dSrc, *dDst;
    size_t nb = (size_t)n*sizeof(double);
    if (alloc_copy_free(src, (void**)&dSrc, nb)) return -1;
    CUDA_CHECK(cudaMalloc((void**)&dDst, nb));
    log_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dSrc, dDst, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, nb, cudaMemcpyDeviceToHost));
    cudaFree(dSrc); cudaFree(dDst);
    return 0;
}

int cuda_softmax(const double* src, double* dst, int num_rows, int dim_size) {
    double *dSrc, *dDst;
    size_t nb = (size_t)num_rows*dim_size*sizeof(double);
    if (alloc_copy_free(src, (void**)&dSrc, nb)) return -1;
    CUDA_CHECK(cudaMalloc((void**)&dDst, nb));
    int tgs = dim_size < BLOCK_SIZE ? dim_size : BLOCK_SIZE;
    int warps = (tgs + 31) / 32;
    softmax_kernel<<<num_rows, tgs, warps*sizeof(double)>>>(dSrc, dDst, dim_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, nb, cudaMemcpyDeviceToHost));
    cudaFree(dSrc); cudaFree(dDst);
    return 0;
}

int cuda_layernorm(const double* src, double* dst,
                   const double* weight, const double* bias,
                   int num_rows, int d_model, double eps) {
    double *dSrc, *dDst, *dW, *dB;
    size_t nb  = (size_t)num_rows*d_model*sizeof(double);
    size_t nb1 = (size_t)d_model*sizeof(double);
    if (alloc_copy_free(src,    (void**)&dSrc, nb))  return -1;
    if (alloc_copy_free(weight, (void**)&dW,   nb1)) { cudaFree(dSrc); return -1; }
    if (alloc_copy_free(bias,   (void**)&dB,   nb1)) { cudaFree(dSrc); cudaFree(dW); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dDst, nb));
    int tgs = d_model < BLOCK_SIZE ? d_model : BLOCK_SIZE;
    int warps = (tgs + 31) / 32;
    layernorm_kernel<<<num_rows, tgs, warps*sizeof(double)>>>(dSrc, dDst, dW, dB, d_model, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, nb, cudaMemcpyDeviceToHost));
    cudaFree(dSrc); cudaFree(dDst); cudaFree(dW); cudaFree(dB);
    return 0;
}

int cuda_rmsnorm(const double* src, double* dst,
                 const double* weight,
                 int num_rows, int d_model, double eps) {
    double *dSrc, *dDst, *dW;
    size_t nb  = (size_t)num_rows*d_model*sizeof(double);
    size_t nb1 = (size_t)d_model*sizeof(double);
    if (alloc_copy_free(src,    (void**)&dSrc, nb))  return -1;
    if (alloc_copy_free(weight, (void**)&dW,   nb1)) { cudaFree(dSrc); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dDst, nb));
    int tgs = d_model < BLOCK_SIZE ? d_model : BLOCK_SIZE;
    int warps = (tgs + 31) / 32;
    rmsnorm_kernel<<<num_rows, tgs, warps*sizeof(double)>>>(dSrc, dDst, dW, d_model, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, nb, cudaMemcpyDeviceToHost));
    cudaFree(dSrc); cudaFree(dDst); cudaFree(dW);
    return 0;
}

} // extern "C"

__global__ void sign_kernel(const double* src, double* dst, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) {
        double v = src[i];
        dst[i] = (v > 0.0) ? 1.0 : (v < 0.0) ? -1.0 : 0.0;
    }
}

__global__ void outer_kernel(const double* a, const double* b, double* dst, int M, int N) {
    int row = blockIdx.y;   // grid.y = M, one block per row
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (col < N) dst[row * N + col] = a[row] * b[col];
}

extern "C" {

int cuda_sign(const double* src, double* dst, int n) {
    double *dSrc, *dDst;
    size_t nb = (size_t)n * sizeof(double);
    if (alloc_copy_free(src, (void**)&dSrc, nb)) return -1;
    CUDA_CHECK(cudaMalloc((void**)&dDst, nb));
    sign_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dSrc, dDst, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, nb, cudaMemcpyDeviceToHost));
    cudaFree(dSrc); cudaFree(dDst);
    return 0;
}

int cuda_outer(const double* a, const double* b, double* dst, int M, int N) {
    double *dA, *dB, *dDst;
    size_t ab = (size_t)M * sizeof(double);
    size_t bb = (size_t)N * sizeof(double);
    size_t db = (size_t)M * N * sizeof(double);
    if (alloc_copy_free(a, (void**)&dA, ab)) return -1;
    if (alloc_copy_free(b, (void**)&dB, bb)) { cudaFree(dA); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dDst, db));
    dim3 block(BLOCK_SIZE, 1);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, M);
    outer_kernel<<<grid, block>>>(dA, dB, dDst, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, db, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dDst);
    return 0;
}

} // extern "C"

// ---------------------------------------------------------------------------
// Optimizer kernels
// ---------------------------------------------------------------------------

// axpy: dst[i] += scale * src[i]
__global__ void axpy_kernel(double* dst, const double* src, double scale, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) dst[i] += scale * src[i];
}

// scale: dst[i] *= s
__global__ void scale_kernel(double* dst, double s, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) dst[i] *= s;
}

// sqrt_vec: dst[i] = sqrt(src[i])
__global__ void sqrt_vec_kernel(const double* src, double* dst, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) dst[i] = sqrt(src[i]);
}

// add_scalar: dst[i] += scalar
__global__ void add_scalar_kernel(double* dst, double scalar, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) dst[i] += scalar;
}

// div_vec: dst[i] = a[i] / b[i]
__global__ void div_vec_kernel(const double* a, const double* b, double* dst, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) dst[i] = a[i] / b[i];
}

// clamp_vec: dst[i] = clamp(dst[i], lo, hi)
__global__ void clamp_vec_kernel(double* dst, double lo, double hi, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) dst[i] = fmax(lo, fmin(hi, dst[i]));
}

extern "C" {

int cuda_axpy(double* dst, const double* src, double scale, int n) {
    double *dDst, *dSrc;
    size_t nb = (size_t)n * sizeof(double);
    if (alloc_copy_free(dst, (void**)&dDst, nb)) return -1;
    if (alloc_copy_free(src, (void**)&dSrc, nb)) { cudaFree(dDst); return -1; }
    axpy_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dDst, dSrc, scale, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, nb, cudaMemcpyDeviceToHost));
    cudaFree(dDst); cudaFree(dSrc);
    return 0;
}

int cuda_scale(double* dst, double s, int n) {
    double *dDst;
    size_t nb = (size_t)n * sizeof(double);
    if (alloc_copy_free(dst, (void**)&dDst, nb)) return -1;
    scale_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dDst, s, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, nb, cudaMemcpyDeviceToHost));
    cudaFree(dDst);
    return 0;
}

int cuda_sqrt_vec(const double* src, double* dst, int n) {
    double *dSrc, *dDst;
    size_t nb = (size_t)n * sizeof(double);
    if (alloc_copy_free(src, (void**)&dSrc, nb)) return -1;
    CUDA_CHECK(cudaMalloc((void**)&dDst, nb));
    sqrt_vec_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dSrc, dDst, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, nb, cudaMemcpyDeviceToHost));
    cudaFree(dSrc); cudaFree(dDst);
    return 0;
}

int cuda_add_scalar(double* dst, double scalar, int n) {
    double *dDst;
    size_t nb = (size_t)n * sizeof(double);
    if (alloc_copy_free(dst, (void**)&dDst, nb)) return -1;
    add_scalar_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dDst, scalar, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, nb, cudaMemcpyDeviceToHost));
    cudaFree(dDst);
    return 0;
}

int cuda_div_vec(const double* a, const double* b, double* dst, int n) {
    double *dA, *dB, *dDst;
    size_t nb = (size_t)n * sizeof(double);
    if (alloc_copy_free(a, (void**)&dA, nb)) return -1;
    if (alloc_copy_free(b, (void**)&dB, nb)) { cudaFree(dA); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dDst, nb));
    div_vec_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dA, dB, dDst, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, nb, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dDst);
    return 0;
}

int cuda_clamp_vec(double* dst, double lo, double hi, int n) {
    double *dDst;
    size_t nb = (size_t)n * sizeof(double);
    if (alloc_copy_free(dst, (void**)&dDst, nb)) return -1;
    clamp_vec_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dDst, lo, hi, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, nb, cudaMemcpyDeviceToHost));
    cudaFree(dDst);
    return 0;
}

} // extern "C"
