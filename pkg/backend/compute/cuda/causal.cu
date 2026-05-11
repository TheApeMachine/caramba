#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include "causal.h"

#define TILE_SIZE  16
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) return -1; \
} while(0)

// ---------------------------------------------------------------------------
// Helper: alloc and copy host→device
// ---------------------------------------------------------------------------
static int alloc_copy(const void* h_src, void** d_ptr, size_t bytes) {
    if (cudaMalloc(d_ptr, bytes) != cudaSuccess) return -1;
    if (cudaMemcpy(*d_ptr, h_src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*d_ptr); return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Tiled matmul kernel: C [M x N] = A [M x K] @ B [K x N]
// ---------------------------------------------------------------------------
__global__ void causal_matmul_kernel(
    const double* __restrict__ A,
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

        for (int i = 0; i < TILE_SIZE; i++) {
            acc += tA[threadIdx.y][i] * tB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Transposed matmul kernel: C [M x N] = A^T [K x M] @ B [K x N]
// Treats A as [K x M] stored row-major, computes A^T [M x K] @ B [K x N].
// ---------------------------------------------------------------------------
__global__ void causal_matmul_t_kernel(
    const double* __restrict__ A,
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
        int aRow = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;
        // A is [K x M]; A^T[row, aRow] = A[aRow * M + row]
        tA[threadIdx.y][threadIdx.x] = (row < M && aRow < K) ? A[aRow * M + row] : 0.0;
        tB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            acc += tA[threadIdx.y][i] * tB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// AXPY kernel: dst[i] += scale * src[i]
// ---------------------------------------------------------------------------
__global__ void causal_axpy_kernel(double* dst, const double* src, double scale, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) dst[i] += scale * src[i];
}

// ---------------------------------------------------------------------------
// Dot product kernel with shared memory reduction
// ---------------------------------------------------------------------------
__global__ void causal_dot_kernel(const double* a, const double* b, double* partial, int n) {
    extern __shared__ double smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = (idx < n) ? a[idx] * b[idx] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = smem[0];
}

// ---------------------------------------------------------------------------
// OLS normal equations helper kernels
// ---------------------------------------------------------------------------

// Compute residual variance: sum((y - X@beta)^2)
__global__ void residual_ss_kernel(
    const double* X, const double* y, const double* beta,
    double* partial, int T, int p)
{
    extern __shared__ double smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double res = 0.0;

    if (idx < T) {
        double pred = 0.0;

        for (int j = 0; j < p; j++) {
            pred += X[idx * p + j] * beta[j];
        }

        double diff = y[idx] - pred;
        res = diff * diff;
    }

    smem[tid] = res;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = smem[0];
}

// Gaussian log probability kernel: log N(x; mu, sigma2)
__global__ void gaussian_logp_kernel(
    const double* x, double mu, double sigma2, double* out, int n)
{
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (i < n) {
        double diff = x[i] - mu;
        out[i] = -0.5 * log(2.0 * M_PI * sigma2) - 0.5 * diff * diff / sigma2;
    }
}

// ---------------------------------------------------------------------------
// Device-side small matrix inverse (Cholesky, max 8x8)
// Runs on a single thread.
// ---------------------------------------------------------------------------
__device__ void chol_invert(const double* A, double* inv, int n) {
    double L[64] = {0};

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = A[i * n + j];

            for (int k = 0; k < j; k++) {
                s -= L[i * n + k] * L[j * n + k];
            }

            L[i * n + j] = (i == j) ? sqrt(fmax(s, 1e-10)) : s / L[j * n + j];
        }
    }

    for (int col = 0; col < n; col++) {
        double y[8] = {0};
        y[col] = 1.0;

        for (int i = col; i < n; i++) {
            if (i > col) {
                y[i] = 0;

                for (int k = col; k < i; k++) {
                    y[i] -= L[i * n + k] * y[k];
                }
            }

            y[i] /= L[i * n + i];
        }

        double x[8] = {0};

        for (int i = n - 1; i >= 0; i--) {
            x[i] = y[i];

            for (int k = i + 1; k < n; k++) {
                x[i] -= L[k * n + i] * x[k];
            }

            x[i] /= L[i * n + i];
        }

        for (int i = 0; i < n; i++) {
            inv[i * n + col] = x[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Do-calculus kernel (single-block; N <= BLOCK_SIZE assumed manageable)
// ---------------------------------------------------------------------------
__global__ void do_calculus_kernel(
    const double* cov, const double* mask, const double* values,
    double* out_mean, double* out_cov, int N)
{
    // Single thread does all work for correctness on small N.
    if (threadIdx.x != 0) return;

    // Identify intervened/free sets.
    int interv[64], nInt = 0;
    int freeIdx[64], nFree = 0;

    for (int i = 0; i < N; i++) {
        if (mask[i] != 0.0) { interv[nInt++] = i; }
        else                 { freeIdx[nFree++] = i; }
    }

    // Initialize output.
    for (int i = 0; i < N; i++) out_mean[i] = 0.0;
    for (int i = 0; i < N * N; i++) out_cov[i] = cov[i];

    for (int k = 0; k < nInt; k++) {
        out_mean[interv[k]] = values[interv[k]];
    }

    if (nInt > 0 && nFree > 0) {
        double sigIntInt[64], sigIntIntInv[64];
        double sigFreeInt[64], sigFreeFree[64], sigIntFree[64];
        double xInt[8], tmp[8], deltaFree[8];

        for (int r = 0; r < nInt; r++)
            for (int c = 0; c < nInt; c++)
                sigIntInt[r * nInt + c] = cov[interv[r] * N + interv[c]];

        chol_invert(sigIntInt, sigIntIntInv, nInt);

        for (int r = 0; r < nFree; r++)
            for (int c = 0; c < nInt; c++)
                sigFreeInt[r * nInt + c] = cov[freeIdx[r] * N + interv[c]];

        for (int k = 0; k < nInt; k++) xInt[k] = values[interv[k]];

        // tmp = sigIntIntInv @ xInt
        for (int r = 0; r < nInt; r++) {
            tmp[r] = 0;
            for (int c = 0; c < nInt; c++) tmp[r] += sigIntIntInv[r * nInt + c] * xInt[c];
        }

        // deltaFree = sigFreeInt @ tmp
        for (int r = 0; r < nFree; r++) {
            deltaFree[r] = 0;
            for (int c = 0; c < nInt; c++) deltaFree[r] += sigFreeInt[r * nInt + c] * tmp[c];
            out_mean[freeIdx[r]] = deltaFree[r];
        }

        for (int r = 0; r < nFree; r++)
            for (int c = 0; c < nFree; c++)
                sigFreeFree[r * nFree + c] = cov[freeIdx[r] * N + freeIdx[c]];

        for (int r = 0; r < nInt; r++)
            for (int c = 0; c < nFree; c++)
                sigIntFree[r * nFree + c] = cov[interv[r] * N + freeIdx[c]];

        // tmp2 = sigIntIntInv @ sigIntFree  [nInt x nFree]
        double tmp2[64];
        for (int r = 0; r < nInt; r++)
            for (int c = 0; c < nFree; c++) {
                tmp2[r * nFree + c] = 0;
                for (int k = 0; k < nInt; k++)
                    tmp2[r * nFree + c] += sigIntIntInv[r * nInt + k] * sigIntFree[k * nFree + c];
            }

        // correction = sigFreeInt @ tmp2  [nFree x nFree]
        double correction[64];
        for (int r = 0; r < nFree; r++)
            for (int c = 0; c < nFree; c++) {
                correction[r * nFree + c] = 0;
                for (int k = 0; k < nInt; k++)
                    correction[r * nFree + c] += sigFreeInt[r * nInt + k] * tmp2[k * nFree + c];
                out_cov[freeIdx[r] * N + freeIdx[c]] = sigFreeFree[r * nFree + c] - correction[r * nFree + c];
            }
    }

    // Zero rows/cols for intervened.
    for (int k = 0; k < nInt; k++) {
        int ii = interv[k];
        for (int j = 0; j < N; j++) {
            out_cov[ii * N + j] = 0.0;
            out_cov[j * N + ii] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// C wrappers
// ---------------------------------------------------------------------------
extern "C" {

int cuda_causal_matmul(const double* A, const double* B, double* C, int M, int K, int N) {
    double *dA, *dB, *dC;
    if (alloc_copy(A, (void**)&dA, (size_t)M*K*sizeof(double))) return -1;
    if (alloc_copy(B, (void**)&dB, (size_t)K*N*sizeof(double))) { cudaFree(dA); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dC, (size_t)M*N*sizeof(double)));
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
    causal_matmul_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(C, dC, (size_t)M*N*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

int cuda_causal_matmul_t(const double* A, const double* B, double* C, int M, int K, int N) {
    // A is [K x M] (treated as transposed), B is [K x N], C is [M x N]
    double *dA, *dB, *dC;
    if (alloc_copy(A, (void**)&dA, (size_t)K*M*sizeof(double))) return -1;
    if (alloc_copy(B, (void**)&dB, (size_t)K*N*sizeof(double))) { cudaFree(dA); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dC, (size_t)M*N*sizeof(double)));
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
    causal_matmul_t_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(C, dC, (size_t)M*N*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

int cuda_causal_axpy(double* dst, const double* src, double scale, int n) {
    double *dDst, *dSrc;
    size_t nb = (size_t)n * sizeof(double);
    if (alloc_copy(dst, (void**)&dDst, nb)) return -1;
    if (alloc_copy(src, (void**)&dSrc, nb)) { cudaFree(dDst); return -1; }
    causal_axpy_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dDst, dSrc, scale, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dst, dDst, nb, cudaMemcpyDeviceToHost));
    cudaFree(dDst); cudaFree(dSrc);
    return 0;
}

int cuda_causal_dot(const double* a, const double* b, double* out, int n) {
    double *dA, *dB, *dPartial;
    size_t nb = (size_t)n * sizeof(double);
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (alloc_copy(a, (void**)&dA, nb)) return -1;
    if (alloc_copy(b, (void**)&dB, nb)) { cudaFree(dA); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dPartial, numBlocks * sizeof(double)));
    causal_dot_kernel<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(double)>>>(dA, dB, dPartial, n);
    CUDA_CHECK(cudaGetLastError());
    double* hPartial = (double*)malloc(numBlocks * sizeof(double));
    if (!hPartial) { cudaFree(dA); cudaFree(dB); cudaFree(dPartial); return -1; }
    CUDA_CHECK(cudaMemcpy(hPartial, dPartial, numBlocks*sizeof(double), cudaMemcpyDeviceToHost));
    double result = 0.0;
    for (int i = 0; i < numBlocks; i++) result += hPartial[i];
    *out = result;
    free(hPartial);
    cudaFree(dA); cudaFree(dB); cudaFree(dPartial);
    return 0;
}

int cuda_causal_do_calculus(
    const double* cov, const double* mask, const double* values,
    double* out, int N)
{
    double *dCov, *dMask, *dValues, *dMean, *dCovOut;
    size_t nb = (size_t)N * sizeof(double);
    size_t nn = (size_t)N * N * sizeof(double);
    if (alloc_copy(cov,    (void**)&dCov,    nn)) return -1;
    if (alloc_copy(mask,   (void**)&dMask,   nb)) { cudaFree(dCov); return -1; }
    if (alloc_copy(values, (void**)&dValues, nb)) { cudaFree(dCov); cudaFree(dMask); return -1; }
    CUDA_CHECK(cudaMalloc((void**)&dMean,   nb));
    CUDA_CHECK(cudaMalloc((void**)&dCovOut, nn));
    do_calculus_kernel<<<1, 1>>>(dCov, dMask, dValues, dMean, dCovOut, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out,     dMean,   nb, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out + N, dCovOut, nn, cudaMemcpyDeviceToHost));
    cudaFree(dCov); cudaFree(dMask); cudaFree(dValues);
    cudaFree(dMean); cudaFree(dCovOut);
    return 0;
}

int cuda_causal_backdoor(
    const double* Y, const double* X, const double* Z,
    double* effect,
    int T, int ny, int nx, int nz)
{
    int p = nx + nz;
    size_t designSz = (size_t)T * p * sizeof(double);
    double* design = (double*)malloc(designSz);
    if (!design) return -1;

    for (int t = 0; t < T; t++) {
        for (int j = 0; j < nx; j++) design[t*p+j] = X[t*nx+j];
        for (int j = 0; j < nz; j++) design[t*p+nx+j] = Z[t*nz+j];
    }

    // W^T W [p x p]
    double* wtw = (double*)calloc(p * p, sizeof(double));
    if (!wtw) { free(design); return -1; }
    cuda_causal_matmul_t(design, design, wtw, p, T, p);

    // Invert W^T W on host (small matrix)
    double* wtwInv = (double*)malloc(p * p * sizeof(double));
    if (!wtwInv) { free(design); free(wtw); return -1; }

    // Cholesky inversion host-side
    double* L = (double*)calloc(p * p, sizeof(double));
    for (int i = 0; i < p; i++) {
        for (int j = 0; j <= i; j++) {
            double s = wtw[i*p+j];
            for (int k = 0; k < j; k++) s -= L[i*p+k]*L[j*p+k];
            L[i*p+j] = (i == j) ? sqrt(fmax(s, 1e-10)) : s / L[j*p+j];
        }
    }
    for (int col = 0; col < p; col++) {
        double y[256] = {0}, x[256] = {0};
        y[col] = 1.0;
        for (int i = col; i < p; i++) {
            if (i > col) { y[i] = 0; for (int k=col; k<i; k++) y[i] -= L[i*p+k]*y[k]; }
            y[i] /= L[i*p+i];
        }
        for (int i = p-1; i >= 0; i--) {
            x[i] = y[i];
            for (int k = i+1; k < p; k++) x[i] -= L[k*p+i]*x[k];
            x[i] /= L[i*p+i];
        }
        for (int i = 0; i < p; i++) wtwInv[i*p+col] = x[i];
    }
    free(L);

    for (int yd = 0; yd < ny; yd++) {
        double* yCol = (double*)malloc(T * sizeof(double));
        for (int t = 0; t < T; t++) yCol[t] = Y[t*ny+yd];

        // W^T y [p]
        double* wty = (double*)calloc(p, sizeof(double));
        for (int t = 0; t < T; t++)
            for (int j = 0; j < p; j++)
                wty[j] += design[t*p+j] * yCol[t];

        // beta = wtwInv @ wty [p]
        double* beta = (double*)calloc(p, sizeof(double));
        for (int i = 0; i < p; i++)
            for (int j = 0; j < p; j++)
                beta[i] += wtwInv[i*p+j] * wty[j];

        double eff = 0.0;
        for (int j = 0; j < nx; j++) eff += beta[j];
        effect[yd] = eff / (double)nx;

        free(yCol); free(wty); free(beta);
    }

    free(design); free(wtw); free(wtwInv);
    return 0;
}

int cuda_causal_iv(
    const double* Z, const double* X, const double* Y,
    double* beta_iv,
    int T, int nz, int nx, int ny)
{
    size_t zSz = (size_t)T*nz*sizeof(double);
    size_t xSz = (size_t)T*nx*sizeof(double);
    size_t ySz = (size_t)T*ny*sizeof(double);

    // Z^T Z [nz x nz]
    double* ztZ = (double*)calloc(nz*nz, sizeof(double));
    cuda_causal_matmul_t(Z, Z, ztZ, nz, T, nz);

    // (Z^T Z)^{-1}
    double* ztZInv = (double*)malloc(nz*nz*sizeof(double));
    {
        double* L = (double*)calloc(nz*nz, sizeof(double));
        for (int i = 0; i < nz; i++) {
            for (int j = 0; j <= i; j++) {
                double s = ztZ[i*nz+j];
                for (int k = 0; k < j; k++) s -= L[i*nz+k]*L[j*nz+k];
                L[i*nz+j] = (i==j) ? sqrt(fmax(s,1e-10)) : s/L[j*nz+j];
            }
        }
        for (int col = 0; col < nz; col++) {
            double y[64]={0}, x[64]={0};
            y[col]=1.0;
            for (int i=col; i<nz; i++) {
                if (i>col) { y[i]=0; for (int k=col;k<i;k++) y[i]-=L[i*nz+k]*y[k]; }
                y[i]/=L[i*nz+i];
            }
            for (int i=nz-1;i>=0;i--) {
                x[i]=y[i];
                for (int k=i+1;k<nz;k++) x[i]-=L[k*nz+i]*x[k];
                x[i]/=L[i*nz+i];
            }
            for (int i=0;i<nz;i++) ztZInv[i*nz+col]=x[i];
        }
        free(L);
    }

    // Z^T X [nz x nx]
    double* ztX = (double*)calloc(nz*nx, sizeof(double));
    cuda_causal_matmul_t(Z, X, ztX, nz, T, nx);

    // proj = (Z^T Z)^{-1} Z^T X [nz x nx]
    double* proj = (double*)calloc(nz*nx, sizeof(double));
    cuda_causal_matmul(ztZInv, ztX, proj, nz, nz, nx);

    // X_hat = Z @ proj [T x nx]
    double* xHat = (double*)calloc(T*nx, sizeof(double));
    cuda_causal_matmul(Z, proj, xHat, T, nz, nx);

    // X_hat^T X_hat [nx x nx]
    double* xhTxh = (double*)calloc(nx*nx, sizeof(double));
    cuda_causal_matmul_t(xHat, xHat, xhTxh, nx, T, nx);

    double* xhTxhInv = (double*)malloc(nx*nx*sizeof(double));
    {
        double* L = (double*)calloc(nx*nx, sizeof(double));
        for (int i=0; i<nx; i++) {
            for (int j=0; j<=i; j++) {
                double s = xhTxh[i*nx+j];
                for (int k=0;k<j;k++) s-=L[i*nx+k]*L[j*nx+k];
                L[i*nx+j]=(i==j)?sqrt(fmax(s,1e-10)):s/L[j*nx+j];
            }
        }
        for (int col=0;col<nx;col++) {
            double y[64]={0},x[64]={0};
            y[col]=1.0;
            for (int i=col;i<nx;i++) {
                if (i>col) { y[i]=0; for (int k=col;k<i;k++) y[i]-=L[i*nx+k]*y[k]; }
                y[i]/=L[i*nx+i];
            }
            for (int i=nx-1;i>=0;i--) {
                x[i]=y[i];
                for (int k=i+1;k<nx;k++) x[i]-=L[k*nx+i]*x[k];
                x[i]/=L[i*nx+i];
            }
            for (int i=0;i<nx;i++) xhTxhInv[i*nx+col]=x[i];
        }
        free(L);
    }

    // X_hat^T Y [nx x ny]
    double* xhTy = (double*)calloc(nx*ny, sizeof(double));
    cuda_causal_matmul_t(xHat, Y, xhTy, nx, T, ny);

    // beta_iv = xhTxhInv @ xhTy [nx x ny]
    cuda_causal_matmul(xhTxhInv, xhTy, beta_iv, nx, nx, ny);

    free(ztZ); free(ztZInv); free(ztX); free(proj);
    free(xHat); free(xhTxh); free(xhTxhInv); free(xhTy);
    return 0;
}

int cuda_causal_cate(
    const double* X, const double* treatment, const double* Y,
    double* cate,
    int T, int nx)
{
    // Count treated/control.
    int n1 = 0, n0 = 0;
    for (int t = 0; t < T; t++) {
        if (treatment[t] >= 0.5) n1++;
        else n0++;
    }

    double* X1 = (double*)malloc(n1*nx*sizeof(double));
    double* Y1 = (double*)malloc(n1*sizeof(double));
    double* X0 = (double*)malloc(n0*nx*sizeof(double));
    double* Y0 = (double*)malloc(n0*sizeof(double));
    int i1 = 0, i0 = 0;

    for (int t = 0; t < T; t++) {
        if (treatment[t] >= 0.5) {
            memcpy(X1+i1*nx, X+t*nx, nx*sizeof(double));
            Y1[i1++] = Y[t];
        } else {
            memcpy(X0+i0*nx, X+t*nx, nx*sizeof(double));
            Y0[i0++] = Y[t];
        }
    }

    // OLS helper: beta = (X^T X)^{-1} X^T y
    auto ols = [&](const double* Xsub, const double* ySub, int n, double* beta) {
        double* xtx = (double*)calloc(nx*nx, sizeof(double));
        cuda_causal_matmul_t(Xsub, Xsub, xtx, nx, n, nx);
        double* xtxInv = (double*)malloc(nx*nx*sizeof(double));
        double* L = (double*)calloc(nx*nx, sizeof(double));
        for (int i=0;i<nx;i++) {
            for (int j=0;j<=i;j++) {
                double s=xtx[i*nx+j];
                for (int k=0;k<j;k++) s-=L[i*nx+k]*L[j*nx+k];
                L[i*nx+j]=(i==j)?sqrt(fmax(s,1e-10)):s/L[j*nx+j];
            }
        }
        for (int col=0;col<nx;col++) {
            double y[64]={0},x[64]={0};
            y[col]=1.0;
            for (int i=col;i<nx;i++) {
                if (i>col) { y[i]=0; for (int k=col;k<i;k++) y[i]-=L[i*nx+k]*y[k]; }
                y[i]/=L[i*nx+i];
            }
            for (int i=nx-1;i>=0;i--) {
                x[i]=y[i];
                for (int k=i+1;k<nx;k++) x[i]-=L[k*nx+i]*x[k];
                x[i]/=L[i*nx+i];
            }
            for (int i=0;i<nx;i++) xtxInv[i*nx+col]=x[i];
        }
        free(L);
        double* xty = (double*)calloc(nx, sizeof(double));
        for (int t=0;t<n;t++)
            for (int j=0;j<nx;j++)
                xty[j] += Xsub[t*nx+j]*ySub[t];
        for (int i=0;i<nx;i++) {
            beta[i]=0;
            for (int j=0;j<nx;j++) beta[i]+=xtxInv[i*nx+j]*xty[j];
        }
        free(xtx); free(xtxInv); free(xty);
    };

    double* beta1 = (double*)calloc(nx, sizeof(double));
    double* beta0 = (double*)calloc(nx, sizeof(double));
    if (n1 > 0) ols(X1, Y1, n1, beta1);
    if (n0 > 0) ols(X0, Y0, n0, beta0);

    for (int t = 0; t < T; t++) {
        double mu1 = 0, mu0 = 0;
        for (int j = 0; j < nx; j++) {
            mu1 += beta1[j] * X[t*nx+j];
            mu0 += beta0[j] * X[t*nx+j];
        }
        cate[t] = mu1 - mu0;
    }

    free(X1); free(Y1); free(X0); free(Y0);
    free(beta1); free(beta0);
    return 0;
}

int cuda_causal_dag_markov(
    const double* X, const double* adj,
    double* log_prob,
    int T, int N)
{
    for (int t = 0; t < T; t++) log_prob[t] = 0.0;

    for (int node = 0; node < N; node++) {
        int parents[64], np = 0;
        for (int j = 0; j < N; j++)
            if (adj[node*N+j] != 0.0) parents[np++] = j;

        double* nodeVals = (double*)malloc(T*sizeof(double));
        for (int t=0;t<T;t++) nodeVals[t] = X[t*N+node];

        double mu = 0.0, sigma2 = 0.0;
        double* beta = (double*)calloc(np, sizeof(double));

        if (np == 0) {
            for (int t=0;t<T;t++) mu += nodeVals[t];
            mu /= T;
            for (int t=0;t<T;t++) { double d=nodeVals[t]-mu; sigma2+=d*d; }
            sigma2 /= T;
            if (sigma2 < 1e-10) sigma2 = 1e-10;
        } else {
            double* pMat = (double*)malloc(T*np*sizeof(double));
            for (int t=0;t<T;t++)
                for (int p=0;p<np;p++)
                    pMat[t*np+p] = X[t*N+parents[p]];

            double* xtx = (double*)calloc(np*np, sizeof(double));
            cuda_causal_matmul_t(pMat, pMat, xtx, np, T, np);
            double* xtxInv = (double*)malloc(np*np*sizeof(double));
            double* L = (double*)calloc(np*np, sizeof(double));
            for (int i=0;i<np;i++) {
                for (int j=0;j<=i;j++) {
                    double s=xtx[i*np+j];
                    for (int k=0;k<j;k++) s-=L[i*np+k]*L[j*np+k];
                    L[i*np+j]=(i==j)?sqrt(fmax(s,1e-10)):s/L[j*np+j];
                }
            }
            for (int col=0;col<np;col++) {
                double y[64]={0},x[64]={0};
                y[col]=1.0;
                for (int i=col;i<np;i++) {
                    if (i>col) { y[i]=0; for (int k=col;k<i;k++) y[i]-=L[i*np+k]*y[k]; }
                    y[i]/=L[i*np+i];
                }
                for (int i=np-1;i>=0;i--) {
                    x[i]=y[i];
                    for (int k=i+1;k<np;k++) x[i]-=L[k*np+i]*x[k];
                    x[i]/=L[i*np+i];
                }
                for (int i=0;i<np;i++) xtxInv[i*np+col]=x[i];
            }
            free(L);
            double* xty = (double*)calloc(np, sizeof(double));
            for (int t=0;t<T;t++)
                for (int p=0;p<np;p++)
                    xty[p] += pMat[t*np+p]*nodeVals[t];
            for (int i=0;i<np;i++) {
                beta[i]=0;
                for (int j=0;j<np;j++) beta[i]+=xtxInv[i*np+j]*xty[j];
            }
            free(xty); free(xtx); free(xtxInv);

            double rss = 0.0;
            for (int t=0;t<T;t++) {
                double pred=0;
                for (int p=0;p<np;p++) pred+=beta[p]*pMat[t*np+p];
                double d=nodeVals[t]-pred;
                rss+=d*d;
            }
            sigma2 = rss / T;
            if (sigma2 < 1e-10) sigma2 = 1e-10;
            free(pMat);
        }

        for (int t=0;t<T;t++) {
            double pred;
            if (np == 0) {
                pred = mu;
            } else {
                pred = 0;
                for (int p=0;p<np;p++) pred += beta[p]*X[t*N+parents[p]];
            }
            double diff = nodeVals[t] - pred;
            log_prob[t] += -0.5*log(2.0*M_PI*sigma2) - 0.5*diff*diff/sigma2;
        }

        free(nodeVals); free(beta);
    }

    return 0;
}

} // extern "C"
