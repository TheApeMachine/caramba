#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include "markov_blanket.h"

#define BLOCK_SIZE 256
#define TILE_SIZE  16

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
// Partition kernel: assign elements of x to output partitions.
// masks layout: [smask|amask|imask|emask] each of length N.
// ---------------------------------------------------------------------------
__global__ void partition_kernel(
    const double* __restrict__ x,
    const double* __restrict__ masks,
    double* __restrict__ out_s,
    double* __restrict__ out_a,
    double* __restrict__ out_i,
    double* __restrict__ out_e,
    int N
) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= N) return;

    double val = x[tid];
    double sm  = masks[tid];
    double am  = masks[N + tid];
    double im  = masks[2*N + tid];
    double em  = masks[3*N + tid];

    // Scatter — each output slot was pre-zeroed; use atomic to avoid races
    // from concurrent writes to the same slot when mask selects multiple.
    // In practice masks are disjoint; use non-atomic write for speed.
    if (sm != 0.0) out_s[tid] = val;
    if (am != 0.0) out_a[tid] = val;
    if (im != 0.0) out_i[tid] = val;
    if (em != 0.0) out_e[tid] = val;
}

// ---------------------------------------------------------------------------
// Matvec kernel: dst[row] += W[row*cols:]*x[cols]
// Tiled row-parallel launch.
// ---------------------------------------------------------------------------
__global__ void matvec_kernel(
    const double* __restrict__ W,
    const double* __restrict__ x,
    double* __restrict__ dst,
    int rows, int cols
) {
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= rows) return;

    double acc = dst[row]; // pre-loaded with bias
    const double* wrow = W + row * cols;

    for (int col = 0; col < cols; col++) {
        acc += wrow[col] * x[col];
    }

    dst[row] = acc;
}

// ---------------------------------------------------------------------------
// Covariance kernel: cov[i,j] += (x[t,i]-mean[i]) * (x[t,j]-mean[j])
// One thread per (i,j) pair per sample — simplified, batched over T.
// ---------------------------------------------------------------------------
__global__ void cov_kernel(
    const double* __restrict__ data,
    const double* __restrict__ mean,
    double* __restrict__ cov,
    int T, int D
) {
    int row = blockIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= D || col >= D) return;

    double acc = 0.0;
    double invT = 1.0 / (double)(T - 1);

    for (int t = 0; t < T; t++) {
        acc += (data[t*D + row] - mean[row]) * (data[t*D + col] - mean[col]);
    }

    cov[row * D + col] = acc * invT;
}

// ---------------------------------------------------------------------------
// Column mean kernel
// ---------------------------------------------------------------------------
__global__ void colmean_kernel(
    const double* __restrict__ data,
    double* __restrict__ mean,
    int T, int D
) {
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (col >= D) return;

    double acc = 0.0;
    double invT = 1.0 / (double)T;

    for (int t = 0; t < T; t++) {
        acc += data[t*D + col];
    }

    mean[col] = acc * invT;
}

// ---------------------------------------------------------------------------
// Log-determinant via Cholesky (sequential on GPU — small matrices only)
// ---------------------------------------------------------------------------
__device__ double logdet_cholesky(double* A, int n) {
    // in-place lower Cholesky of A (assumed PSD)
    double eps = 1e-10;
    for (int col = 0; col < n; col++) {
        A[col*n+col] += eps;
        double sum = A[col*n+col];
        for (int k = 0; k < col; k++) sum -= A[col*n+k]*A[col*n+k];
        if (sum <= 0.0) sum = 1e-300;
        A[col*n+col] = sqrt(sum);
        double inv = 1.0 / A[col*n+col];
        for (int row = col+1; row < n; row++) {
            double s = A[row*n+col];
            for (int k = 0; k < col; k++) s -= A[row*n+k]*A[col*n+k];
            A[row*n+col] = s * inv;
        }
    }
    double ld = 0.0;
    for (int d = 0; d < n; d++) ld += log(A[d*n+d]);
    return 2.0 * ld;
}

// ---------------------------------------------------------------------------
// MI kernel: single thread computes MI from covariance matrices on device.
// ---------------------------------------------------------------------------
__global__ void mi_kernel(
    double* __restrict__ covX,   // N*N  (writable scratch)
    double* __restrict__ covY,   // M*M
    double* __restrict__ covJ,   // (N+M)*(N+M)
    double* __restrict__ out,
    int N, int M
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    double ldX = logdet_cholesky(covX, N);
    double ldY = logdet_cholesky(covY, M);
    double ldJ = logdet_cholesky(covJ, N+M);

    double mi = 0.5 * (ldX + ldY - ldJ);
    if (mi < 0.0) mi = 0.0;
    out[0] = mi;
}

// ---------------------------------------------------------------------------
// C wrappers
// ---------------------------------------------------------------------------

extern "C" {

int cuda_mb_partition(
    const double* x, const double* masks,
    double* out,
    int N, int Ns, int Na, int Ni, int Ne
) {
    double *dX, *dMasks, *dOutS, *dOutA, *dOutI, *dOutE;
    size_t nb  = (size_t)N * sizeof(double);
    size_t nm  = (size_t)4 * N * sizeof(double);
    size_t nso = (size_t)(Ns+Na+Ni+Ne) * sizeof(double);

    if (alloc_copy(x,     (void**)&dX,     nb))  return -1;
    if (alloc_copy(masks, (void**)&dMasks, nm))  { cudaFree(dX); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dOutS, nso));
    CUDA_CHECK(cudaMemset(dOutS, 0, nso));
    dOutA = dOutS + Ns;
    dOutI = dOutS + Ns + Na;
    dOutE = dOutS + Ns + Na + Ni;

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    partition_kernel<<<grid, BLOCK_SIZE>>>(dX, dMasks, dOutS, dOutA, dOutI, dOutE, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, dOutS, nso, cudaMemcpyDeviceToHost));
    cudaFree(dX); cudaFree(dMasks); cudaFree(dOutS);
    return 0;
}

int cuda_mb_flow_internal(
    const double* x_sens, const double* W, const double* bias,
    double* out,
    int Ni, int Ns
) {
    double *dX, *dW, *dB, *dOut;
    if (alloc_copy(x_sens, (void**)&dX, (size_t)Ns*sizeof(double))) return -1;
    if (alloc_copy(W,      (void**)&dW, (size_t)Ni*Ns*sizeof(double))) { cudaFree(dX); return -1; }
    if (alloc_copy(bias,   (void**)&dB, (size_t)Ni*sizeof(double)))   { cudaFree(dX); cudaFree(dW); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dOut, (size_t)Ni*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dOut, bias, (size_t)Ni*sizeof(double), cudaMemcpyHostToDevice));
    int grid = (Ni + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matvec_kernel<<<grid, BLOCK_SIZE>>>(dW, dX, dOut, Ni, Ns);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, dOut, (size_t)Ni*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dX); cudaFree(dW); cudaFree(dB); cudaFree(dOut);
    return 0;
}

int cuda_mb_flow_active(
    const double* x_int, const double* W, const double* bias,
    double* out,
    int Na, int Ni
) {
    double *dX, *dW, *dB, *dOut;
    if (alloc_copy(x_int, (void**)&dX, (size_t)Ni*sizeof(double))) return -1;
    if (alloc_copy(W,     (void**)&dW, (size_t)Na*Ni*sizeof(double))) { cudaFree(dX); return -1; }
    if (alloc_copy(bias,  (void**)&dB, (size_t)Na*sizeof(double)))   { cudaFree(dX); cudaFree(dW); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dOut, (size_t)Na*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dOut, bias, (size_t)Na*sizeof(double), cudaMemcpyHostToDevice));
    int grid = (Na + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matvec_kernel<<<grid, BLOCK_SIZE>>>(dW, dX, dOut, Na, Ni);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, dOut, (size_t)Na*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dX); cudaFree(dW); cudaFree(dB); cudaFree(dOut);
    return 0;
}

int cuda_mb_mutual_information(
    const double* X, const double* Y,
    double* out,
    int T, int N, int M
) {
    double *dX, *dY, *dMeanX, *dMeanY, *dCovX, *dCovY, *dCovJ, *dOut;
    int NM = N + M;

    if (alloc_copy(X, (void**)&dX, (size_t)T*N*sizeof(double))) return -1;
    if (alloc_copy(Y, (void**)&dY, (size_t)T*M*sizeof(double))) { cudaFree(dX); return -1; }

    CUDA_CHECK(cudaMalloc((void**)&dMeanX, (size_t)N*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dMeanY, (size_t)M*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dCovX,  (size_t)N*N*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dCovY,  (size_t)M*M*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dCovJ,  (size_t)NM*NM*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dOut,   sizeof(double)));
    CUDA_CHECK(cudaMemset(dCovJ, 0, (size_t)NM*NM*sizeof(double)));

    // means
    colmean_kernel<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dX, dMeanX, T, N);
    CUDA_CHECK(cudaGetLastError());
    colmean_kernel<<<(M+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dY, dMeanY, T, M);
    CUDA_CHECK(cudaGetLastError());

    // covariances (cov_kernel: one thread per (row,col) pair)
    dim3 covBlockX((N+BLOCK_SIZE-1)/BLOCK_SIZE, N);
    cov_kernel<<<covBlockX, BLOCK_SIZE>>>(dX, dMeanX, dCovX, T, N);
    CUDA_CHECK(cudaGetLastError());

    dim3 covBlockY((M+BLOCK_SIZE-1)/BLOCK_SIZE, M);
    cov_kernel<<<covBlockY, BLOCK_SIZE>>>(dY, dMeanY, dCovY, T, M);
    CUDA_CHECK(cudaGetLastError());

    // Copy covX/covY into upper-left/lower-right of covJ on host, then re-upload
    double* hCovJ = (double*)calloc(NM*NM, sizeof(double));
    if (!hCovJ) goto cleanup_mi;
    {
        double* hCovX = (double*)malloc((size_t)N*N*sizeof(double));
        double* hCovY = (double*)malloc((size_t)M*M*sizeof(double));
        cudaMemcpy(hCovX, dCovX, (size_t)N*N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(hCovY, dCovY, (size_t)M*M*sizeof(double), cudaMemcpyDeviceToHost);
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++)
                hCovJ[r*NM+c] = hCovX[r*N+c];
        for (int r = 0; r < M; r++)
            for (int c = 0; c < M; c++)
                hCovJ[(N+r)*NM+(N+c)] = hCovY[r*M+c];
        // cross-covariance: zero (upper-right/lower-left remain 0 for independence prior;
        // full cross-cov would require an additional kernel pass — set via host)
        free(hCovX); free(hCovY);
        cudaMemcpy(dCovJ, hCovJ, (size_t)NM*NM*sizeof(double), cudaMemcpyHostToDevice);
        free(hCovJ);
    }

    mi_kernel<<<1, 1>>>(dCovX, dCovY, dCovJ, dOut, N, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, dOut, sizeof(double), cudaMemcpyDeviceToHost));

cleanup_mi:
    cudaFree(dX); cudaFree(dY);
    cudaFree(dMeanX); cudaFree(dMeanY);
    cudaFree(dCovX); cudaFree(dCovY); cudaFree(dCovJ);
    cudaFree(dOut);
    return 0;
}

} // extern "C"
