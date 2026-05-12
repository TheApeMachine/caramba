#include <cuda_runtime.h>
#include <math.h>
#include "vsa.h"

#define BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// Helper: alloc device buffer and copy from host
// ---------------------------------------------------------------------------

static int vsa_alloc_copy(const void* h_src, void** d_ptr, size_t bytes) {
    if (cudaMalloc(d_ptr, bytes) != cudaSuccess) return -1;
    if (cudaMemcpy(*d_ptr, h_src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*d_ptr); return -1;
    }
    return 0;
}

#define VSA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) return -1; \
} while(0)

// ---------------------------------------------------------------------------
// bind_kernel: elementwise multiply
// ---------------------------------------------------------------------------

__global__ void vsa_bind_kernel(const double* a, const double* b, double* out, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

// ---------------------------------------------------------------------------
// bundle_sum_kernel: accumulate one source vector into out (called per vec)
// ---------------------------------------------------------------------------

__global__ void vsa_bundle_sum_kernel(double* out, const double* src, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < n) out[idx] += src[idx];
}

// ---------------------------------------------------------------------------
// bundle_normalize_kernel: out[i] *= inv_norm (inv from prior kernel)
// ---------------------------------------------------------------------------

__global__ void vsa_inverse_l2_norm_kernel(const double* sumsq, double* inv_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    double s = sumsq[0];
    double norm = sqrt(fmax(s, 0.0));
    inv_out[0] = (norm > 1e-12) ? (1.0 / norm) : 0.0;
}

__global__ void vsa_normalize_global_inv_kernel(double* out, const double* inv_ptr, int n) {
    double inv = inv_ptr[0];
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < n) out[idx] *= inv;
}

// ---------------------------------------------------------------------------
// sum_sq_kernel: partial sum-of-squares reduction using shared memory
// ---------------------------------------------------------------------------

__global__ void vsa_sum_sq_kernel(const double* a, double* out_sum, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    sdata[tid] = (idx < n) ? a[idx] * a[idx] : 0.0;
    __syncthreads();

    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out_sum, sdata[0]);
}

// ---------------------------------------------------------------------------
// dot_kernel: partial dot-product reduction using shared memory
// ---------------------------------------------------------------------------

__global__ void vsa_dot_kernel(const double* a, const double* b, double* out_sum, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    sdata[tid] = (idx < n) ? a[idx] * b[idx] : 0.0;
    __syncthreads();

    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out_sum, sdata[0]);
}

// ---------------------------------------------------------------------------
// C wrappers
// ---------------------------------------------------------------------------

extern "C" {

int cuda_vsa_bind(const double* a, const double* b, double* out, int n) {
    double *dA, *dB, *dOut;
    size_t nb = (size_t)n * sizeof(double);

    if (vsa_alloc_copy(a, (void**)&dA, nb)) return -1;
    if (vsa_alloc_copy(b, (void**)&dB, nb)) { cudaFree(dA); return -1; }
    VSA_CHECK(cudaMalloc((void**)&dOut, nb));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vsa_bind_kernel<<<blocks, BLOCK_SIZE>>>(dA, dB, dOut, n);
    VSA_CHECK(cudaGetLastError());
    VSA_CHECK(cudaMemcpy(out, dOut, nb, cudaMemcpyDeviceToHost));

    cudaFree(dA); cudaFree(dB); cudaFree(dOut);
    return 0;
}

int cuda_vsa_bundle(const double** vecs, int num_vecs, double* out, int n) {
    size_t nb = (size_t)n * sizeof(double);
    double *dOut;
    VSA_CHECK(cudaMalloc((void**)&dOut, nb));
    VSA_CHECK(cudaMemset(dOut, 0, nb));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int v = 0; v < num_vecs; v++) {
        double* dSrc;
        if (vsa_alloc_copy(vecs[v], (void**)&dSrc, nb)) {
            cudaFree(dOut); return -1;
        }
        vsa_bundle_sum_kernel<<<blocks, BLOCK_SIZE>>>(dOut, dSrc, n);
        cudaFree(dSrc);
        if (cudaGetLastError() != cudaSuccess) { cudaFree(dOut); return -1; }
    }

    // Compute L2 norm via sum-of-squares reduction
    double *dTotal = NULL;
    VSA_CHECK(cudaMalloc((void**)&dTotal, sizeof(double)));
    VSA_CHECK(cudaMemset(dTotal, 0, sizeof(double)));

    vsa_sum_sq_kernel<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(dOut, dTotal, n);
    VSA_CHECK(cudaGetLastError());

    double *dInv = NULL;
    VSA_CHECK(cudaMalloc((void**)&dInv, sizeof(double)));
    vsa_inverse_l2_norm_kernel<<<1, 1>>>(dTotal, dInv);
    VSA_CHECK(cudaGetLastError());

    vsa_normalize_global_inv_kernel<<<blocks, BLOCK_SIZE>>>(dOut, dInv, n);
    VSA_CHECK(cudaGetLastError());
    cudaFree(dTotal);
    cudaFree(dInv);

    VSA_CHECK(cudaMemcpy(out, dOut, nb, cudaMemcpyDeviceToHost));
    cudaFree(dOut);
    return 0;
}

int cuda_vsa_similarity(const double* a, const double* b, double* out, int n) {
    double *dA, *dB;
    size_t nb = (size_t)n * sizeof(double);

    if (vsa_alloc_copy(a, (void**)&dA, nb)) return -1;
    if (vsa_alloc_copy(b, (void**)&dB, nb)) { cudaFree(dA); return -1; }

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double *dSum = NULL;
    VSA_CHECK(cudaMalloc((void**)&dSum, sizeof(double)));
    VSA_CHECK(cudaMemset(dSum, 0, sizeof(double)));

    vsa_dot_kernel<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(dA, dB, dSum, n);
    VSA_CHECK(cudaGetLastError());

    VSA_CHECK(cudaMemcpy(out, dSum, sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dSum);
    cudaFree(dA); cudaFree(dB);
    return 0;
}

} // extern "C"
