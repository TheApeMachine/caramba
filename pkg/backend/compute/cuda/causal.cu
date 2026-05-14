#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
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
__global__ void causal_dot_kernel(const double* a, const double* b, double* out_sum, int n) {
    extern __shared__ double smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = (idx < n) ? a[idx] * b[idx] : 0.0;
    __syncthreads();

    #pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out_sum, smem[0]);
}

__global__ void pack_design_kernel(
    const double* X, const double* Z, double* design,
    int T, int nx, int nz)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    int p = nx + nz;
    double* row = design + t * p;

    for (int j = 0; j < nx; j++) row[j] = X[t * nx + j];

    for (int j = 0; j < nz; j++) row[nx + j] = Z[t * nz + j];
}

__global__ void cate_fill_indices_kernel(
    const double* treatment, int T,
    int* idx1, int* idx0, int* c1, int* c0)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    if (treatment[t] >= 0.5) {
        int k = atomicAdd(c1, 1);
        idx1[k] = t;
    } else {
        int k = atomicAdd(c0, 1);
        idx0[k] = t;
    }
}

__global__ void gather_X_rows_kernel(
    const double* X, const int* idx, int count, int nx, double* out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= count) return;

    int t = idx[r];

    for (int j = 0; j < nx; j++) out[r * nx + j] = X[t * nx + j];
}

__global__ void gather_y_values_kernel(
    const double* y, const int* idx, int count, double* out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= count) return;

    out[r] = y[idx[r]];
}

__global__ void cate_predict_kernel(
    const double* X, const double* b1, const double* b0,
    int T, int nx, double* out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    double m1 = 0.0, m0 = 0.0;

    for (int j = 0; j < nx; j++) {
        double xv = X[t * nx + j];
        m1 += b1[j] * xv;
        m0 += b0[j] * xv;
    }

    out[t] = m1 - m0;
}

__global__ void dag_gather_parents_kernel(
    const double* X, int T, int N,
    const int* parents, int np,
    double* pMat)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    for (int j = 0; j < np; j++) pMat[t * np + j] = X[t * N + parents[j]];
}

__global__ void extract_column_kernel(
    const double* X, int T, int N, int col, double* out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) out[t] = X[t * N + col];
}

__global__ void reduce_mean_var_kernel(const double* v, int T, double* meanOut, double* varOut) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    double s = 0.0;

    for (int i = 0; i < T; i++) s += v[i];

    double mu = s / (double)T;
    double q = 0.0;

    for (int i = 0; i < T; i++) {
        double d = v[i] - mu;
        q += d * d;
    }

    meanOut[0] = mu;
    varOut[0] = fmax(q / (double)T, 1e-10);
}

__global__ void rss_kernel(
    const double* Xmat, const double* y, const double* beta, int T, int p, double* rssOut)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    double rss = 0.0;

    for (int t = 0; t < T; t++) {
        double pred = 0.0;

        for (int j = 0; j < p; j++) pred += Xmat[t * p + j] * beta[j];

        double d = y[t] - pred;
        rss += d * d;
    }

    rssOut[0] = rss;
}

__global__ void dag_conditional_logp_kernel(
    const double* nodeVals, const double* X, int T, int N,
    const int* parents, int np, const double* beta, double sigma2, double* logProb)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    double pred = 0.0;

    for (int j = 0; j < np; j++) {
        pred += beta[j] * X[t * N + parents[j]];
    }

    double diff = nodeVals[t] - pred;
    double lp = -0.5 * log(2.0 * M_PI * sigma2) - 0.5 * diff * diff / sigma2;
    atomicAdd(&logProb[t], lp);
}

__global__ void dag_logp_univariate_kernel(
    const double* nodeVals, double mu, double sigma2, double* logProb, int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    double diff = nodeVals[t] - mu;
    double lp = -0.5 * log(2.0 * M_PI * sigma2) - 0.5 * diff * diff / sigma2;
    atomicAdd(&logProb[t], lp);
}

__global__ void backdoor_effect_kernel(const double* beta, int nx, int ny, double* effect) {
    int yd = blockIdx.x * blockDim.x + threadIdx.x;
    if (yd >= ny) return;

    double sum = 0.0;

    for (int j = 0; j < nx; j++) sum += beta[j * ny + yd];

    effect[yd] = sum / (double)nx;
}

__global__ void counterfactual_kernel(
    const double* x_obs,
    const double* y_obs,
    const double* beta,
    const double* x_cf,
    double* out,
    int n,
    int n_cf)
{
    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = n * n_cf;
    if (index >= total) return;

    int obs_index = index / n_cf;
    int cf_index = index - obs_index * n_cf;
    double epsilon = y_obs[obs_index] - beta[obs_index] * x_obs[obs_index];
    out[index] = beta[obs_index] * x_cf[cf_index] + epsilon;
}

__global__ void fill_quantile_boundaries_kernel(
    const double* sorted,
    double* boundaries,
    int samples,
    int bins)
{
    int boundary_index = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (boundary_index >= bins) return;

    int position = (int)(((double)boundary_index / (double)bins) * (double)samples);
    if (position >= samples) position = samples - 1;
    boundaries[boundary_index - 1] = sorted[position];
}

__global__ void assign_frontdoor_bins_kernel(
    const double* values,
    const double* boundaries,
    int* bins_out,
    int samples,
    int bins)
{
    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (index >= samples) return;

    double value = values[index];
    int bin = 0;

    while (bin < bins - 1 && !(value < boundaries[bin])) {
        bin++;
    }

    bins_out[index] = bin;
}

__global__ void frontdoor_accumulate_kernel(
    const int* x_bins,
    const int* m_bins,
    const double* y,
    double* p_x,
    double* count_x,
    double* p_m_given_x,
    double* e_y_given_xm,
    double* count_xm,
    int samples,
    int nx,
    int nm)
{
    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (index >= samples) return;

    int x_bin = x_bins[index];
    int m_bin = m_bins[index];
    atomicAdd(&p_x[x_bin], 1.0);
    atomicAdd(&count_x[x_bin], 1.0);
    atomicAdd(&p_m_given_x[m_bin * nx + x_bin], 1.0);
    atomicAdd(&e_y_given_xm[x_bin * nm + m_bin], y[index]);
    atomicAdd(&count_xm[x_bin * nm + m_bin], 1.0);
}

__global__ void frontdoor_normalize_kernel(
    double* p_x,
    double* count_x,
    double* p_m_given_x,
    double* e_y_given_xm,
    double* count_xm,
    int samples,
    int nx,
    int nm)
{
    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int p_m_total = nx * nm;

    if (index < nx) {
        p_x[index] /= (double)samples;
    }

    if (index < p_m_total) {
        int x_bin = index % nx;
        if (count_x[x_bin] > 0.0) {
            p_m_given_x[index] /= count_x[x_bin];
        }
    }

    if (index < p_m_total) {
        if (count_xm[index] > 0.0) {
            e_y_given_xm[index] /= count_xm[index];
        } else {
            e_y_given_xm[index] = NAN;
        }
    }
}

__global__ void frontdoor_effect_kernel(
    const double* p_x,
    const double* p_m_given_x,
    const double* e_y_given_xm,
    double* effect,
    int nx,
    int nm)
{
    int x_bin = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x_bin >= nx) return;

    double value = 0.0;

    for (int m_bin = 0; m_bin < nm; m_bin++) {
        double inner = 0.0;

        for (int x_prime = 0; x_prime < nx; x_prime++) {
            double mean = e_y_given_xm[x_prime * nm + m_bin];
            if (!isnan(mean)) {
                inner += mean * p_x[x_prime];
            }
        }

        value += p_m_given_x[m_bin * nx + x_bin] * inner;
    }

    effect[x_bin] = value;
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

    #pragma unroll
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
// Device SPD inverse via Cholesky (row-major, sym A). work: p*p + 2*p doubles.
// ---------------------------------------------------------------------------
__device__ void chol_invert_bufs(
    const double* A, double* inv, int n,
    double* L, double* y, double* x
) {
    for (int i = 0; i < n * n; i++) {
        L[i] = 0.0;
    }

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
        for (int ii = 0; ii < n; ii++) y[ii] = 0.0;

        y[col] = 1.0;

        for (int i = col; i < n; i++) {
            if (i > col) {
                y[i] = 0.0;

                for (int k = col; k < i; k++) {
                    y[i] -= L[i * n + k] * y[k];
                }
            }

            y[i] /= L[i * n + i];
        }

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

__global__ void invert_spd_global_kernel(const double* A, double* inv, int p, double* work) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    double* L = work;
    double* y = L + p * p;
    double* x = y + p;
    chol_invert_bufs(A, inv, p, L, y, x);
}

#define CAUSAL_BLOCK_CAP 64

// ---------------------------------------------------------------------------
// Do-calculus kernel (single thread; scratch = 8 * 64*64 doubles + Cholesky tail).
// ---------------------------------------------------------------------------
__global__ void do_calculus_kernel(
    const double* cov, const double* mask, const double* values,
    double* out_mean, double* out_cov, int N, double* scratch)
{
    if (threadIdx.x != 0) return;

    int interv[CAUSAL_BLOCK_CAP], nInt = 0;
    int freeIdx[CAUSAL_BLOCK_CAP], nFree = 0;

    for (int i = 0; i < N; i++) {
        if (mask[i] != 0.0) { interv[nInt++] = i; }
        else                 { freeIdx[nFree++] = i; }
    }

    for (int i = 0; i < N; i++) out_mean[i] = 0.0;
    for (int i = 0; i < N * N; i++) out_cov[i] = cov[i];

    for (int k = 0; k < nInt; k++) {
        out_mean[interv[k]] = values[interv[k]];
    }

    if (nInt > 0 && nFree > 0) {
        size_t step = (size_t)CAUSAL_BLOCK_CAP * CAUSAL_BLOCK_CAP;
        double* sigIntInt   = scratch;
        double* sigIntIntInv = scratch + step;
        double* sigFreeInt  = scratch + 2 * step;
        double* sigFreeFree = scratch + 3 * step;
        double* sigIntFree  = scratch + 4 * step;
        double* tmp2        = scratch + 5 * step;
        double* correction  = scratch + 6 * step;
        double* cholL       = scratch + 7 * step;
        double* choly       = cholL + step;
        double* cholx       = choly + CAUSAL_BLOCK_CAP;

        double xInt[CAUSAL_BLOCK_CAP], tmp[CAUSAL_BLOCK_CAP], deltaFree[CAUSAL_BLOCK_CAP];

        for (int r = 0; r < nInt; r++) {
            for (int c = 0; c < nInt; c++) {
                sigIntInt[r * nInt + c] = cov[interv[r] * N + interv[c]];
            }
        }

        chol_invert_bufs(sigIntInt, sigIntIntInv, nInt, cholL, choly, cholx);

        for (int r = 0; r < nFree; r++) {
            for (int c = 0; c < nInt; c++) {
                sigFreeInt[r * nInt + c] = cov[freeIdx[r] * N + interv[c]];
            }
        }

        for (int k = 0; k < nInt; k++) xInt[k] = values[interv[k]];

        for (int r = 0; r < nInt; r++) {
            tmp[r] = 0.0;

            for (int c = 0; c < nInt; c++) tmp[r] += sigIntIntInv[r * nInt + c] * xInt[c];
        }

        for (int r = 0; r < nFree; r++) {
            deltaFree[r] = 0.0;

            for (int c = 0; c < nInt; c++) deltaFree[r] += sigFreeInt[r * nInt + c] * tmp[c];

            out_mean[freeIdx[r]] = deltaFree[r];
        }

        for (int r = 0; r < nFree; r++) {
            for (int c = 0; c < nFree; c++) {
                sigFreeFree[r * nFree + c] = cov[freeIdx[r] * N + freeIdx[c]];
            }
        }

        for (int r = 0; r < nInt; r++) {
            for (int c = 0; c < nFree; c++) {
                sigIntFree[r * nFree + c] = cov[interv[r] * N + freeIdx[c]];
            }
        }

        for (int r = 0; r < nInt; r++) {
            for (int c = 0; c < nFree; c++) {
                tmp2[r * nFree + c] = 0.0;

                for (int k = 0; k < nInt; k++) {
                    tmp2[r * nFree + c] += sigIntIntInv[r * nInt + k] * sigIntFree[k * nFree + c];
                }
            }
        }

        for (int r = 0; r < nFree; r++) {
            for (int c = 0; c < nFree; c++) {
                correction[r * nFree + c] = 0.0;

                for (int k = 0; k < nInt; k++) {
                    correction[r * nFree + c] += sigFreeInt[r * nInt + k] * tmp2[k * nFree + c];
                }

                out_cov[freeIdx[r] * N + freeIdx[c]] =
                    sigFreeFree[r * nFree + c] - correction[r * nFree + c];
            }
        }
    }

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

static int causal_launch_matmul_dev(
    const double* dA, const double* dB, double* dC, int M, int K, int N)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    causal_matmul_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

static int causal_launch_matmul_t_dev(
    const double* dA, const double* dB, double* dC, int M, int K, int N)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    causal_matmul_t_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

static int ols_fit_device_dim(
    const double* dXsub, int n, int p, const double* dYcol,
    double* dBeta,
    double* dXTX, double* dXTXInv, double* dXTY,
    double* dInvWork)
{
    if (n <= 0) {
        cudaMemset(dBeta, 0, (size_t)p * sizeof(double));
        return 0;
    }

    if (causal_launch_matmul_t_dev(dXsub, dXsub, dXTX, p, n, p) != 0) return -1;

    invert_spd_global_kernel<<<1, 1>>>(dXTX, dXTXInv, p, dInvWork);

    if (cudaGetLastError() != cudaSuccess) return -1;

    if (causal_launch_matmul_t_dev(dXsub, dYcol, dXTY, p, n, 1) != 0) return -1;

    if (causal_launch_matmul_dev(dXTXInv, dXTY, dBeta, p, p, 1) != 0) return -1;

    return 0;
}

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
    double *dA, *dB, *dSum;
    size_t nb = (size_t)n * sizeof(double);
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (alloc_copy(a, (void**)&dA, nb)) return -1;
    if (alloc_copy(b, (void**)&dB, nb)) { cudaFree(dA); return -1; }

    CUDA_CHECK(cudaMalloc((void**)&dSum, sizeof(double)));
    CUDA_CHECK(cudaMemset(dSum, 0, sizeof(double)));

    causal_dot_kernel<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(double)>>>(dA, dB, dSum, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(out, dSum, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dSum);
    cudaFree(dA); cudaFree(dB);
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
    size_t scratchElems = 8 * (size_t)CAUSAL_BLOCK_CAP * CAUSAL_BLOCK_CAP + 128;
    double* dScratch = NULL;
    CUDA_CHECK(cudaMalloc((void**)&dScratch, scratchElems * sizeof(double)));
    do_calculus_kernel<<<1, 1>>>(dCov, dMask, dValues, dMean, dCovOut, N, dScratch);
    CUDA_CHECK(cudaGetLastError());
    cudaFree(dScratch);
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
    double *dX = NULL, *dZ = NULL, *dY = NULL;
    double *dDesign = NULL, *dWTW = NULL, *dWTWInv = NULL;
    double *dWTY = NULL, *dBeta = NULL, *dEffect = NULL;
    double *dInvWork = NULL;

    if (alloc_copy(X, (void**)&dX, (size_t)T * nx * sizeof(double))) return -1;
    if (alloc_copy(Z, (void**)&dZ, (size_t)T * nz * sizeof(double))) { cudaFree(dX); return -1; }
    if (alloc_copy(Y, (void**)&dY, (size_t)T * ny * sizeof(double))) {
        cudaFree(dX); cudaFree(dZ); return -1;
    }

    CUDA_CHECK(cudaMalloc((void**)&dDesign, (size_t)T * p * sizeof(double)));
    {
        int gridP = (T + BLOCK_SIZE - 1) / BLOCK_SIZE;
        pack_design_kernel<<<gridP, BLOCK_SIZE>>>(dX, dZ, dDesign, T, nx, nz);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMalloc((void**)&dWTW, (size_t)p * p * sizeof(double)));
    if (causal_launch_matmul_t_dev(dDesign, dDesign, dWTW, p, T, p)   != 0) goto fail_backdoor;

    CUDA_CHECK(cudaMalloc((void**)&dWTWInv, (size_t)p * p * sizeof(double)));
    {
        size_t wz = ((size_t)p * p + (size_t)2 * p) * sizeof(double);
        CUDA_CHECK(cudaMalloc((void**)&dInvWork, wz));
        invert_spd_global_kernel<<<1, 1>>>(dWTW, dWTWInv, p, dInvWork);
        CUDA_CHECK(cudaGetLastError());
        cudaFree(dInvWork);
        dInvWork = NULL;
    }

    CUDA_CHECK(cudaMalloc((void**)&dWTY, (size_t)p * ny * sizeof(double)));
    if (causal_launch_matmul_t_dev(dDesign, dY, dWTY, p, T, ny) != 0) goto fail_backdoor;

    CUDA_CHECK(cudaMalloc((void**)&dBeta, (size_t)p * ny * sizeof(double)));
    if (causal_launch_matmul_dev(dWTWInv, dWTY, dBeta, p, p, ny) != 0) goto fail_backdoor;

    CUDA_CHECK(cudaMalloc((void**)&dEffect, (size_t)ny * sizeof(double)));
    {
        int gridE = (ny + BLOCK_SIZE - 1) / BLOCK_SIZE;
        backdoor_effect_kernel<<<gridE, BLOCK_SIZE>>>(dBeta, nx, ny, dEffect);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(effect, dEffect, (size_t)ny * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dX); cudaFree(dZ); cudaFree(dY); cudaFree(dDesign);
    cudaFree(dWTW); cudaFree(dWTWInv); cudaFree(dWTY); cudaFree(dBeta); cudaFree(dEffect);
    return 0;

fail_backdoor:
    if (dInvWork) cudaFree(dInvWork);
    cudaFree(dX); cudaFree(dZ); cudaFree(dY);
    if (dDesign) cudaFree(dDesign);
    if (dWTW) cudaFree(dWTW);
    if (dWTWInv) cudaFree(dWTWInv);
    if (dWTY) cudaFree(dWTY);
    if (dBeta) cudaFree(dBeta);
    if (dEffect) cudaFree(dEffect);
    return -1;
}

int cuda_causal_iv(
    const double* Z, const double* X, const double* Y,
    double* beta_iv,
    int T, int nz, int nx, int ny)
{
    double *dZ = NULL, *dX = NULL, *dY = NULL;
    double *dZtZ = NULL, *dZtZInv = NULL, *dZtX = NULL, *dProj = NULL;
    double *dXHat = NULL, *dXhTXh = NULL, *dXhTXhInv = NULL, *dXhTY = NULL;
    double *dBetaIv = NULL;
    double *dInvWork = NULL;

    if (alloc_copy(Z, (void**)&dZ, (size_t)T * nz * sizeof(double))) return -1;
    if (alloc_copy(X, (void**)&dX, (size_t)T * nx * sizeof(double))) { cudaFree(dZ); return -1; }
    if (alloc_copy(Y, (void**)&dY, (size_t)T * ny * sizeof(double))) {
        cudaFree(dZ); cudaFree(dX); return -1;
    }

    CUDA_CHECK(cudaMalloc((void**)&dZtZ, (size_t)nz * nz * sizeof(double)));
    if (causal_launch_matmul_t_dev(dZ, dZ, dZtZ, nz, T, nz) != 0) goto fail_iv;

    CUDA_CHECK(cudaMalloc((void**)&dZtZInv, (size_t)nz * nz * sizeof(double)));
    {
        size_t wz = ((size_t)nz * nz + (size_t)2 * nz) * sizeof(double);
        CUDA_CHECK(cudaMalloc((void**)&dInvWork, wz));
        invert_spd_global_kernel<<<1, 1>>>(dZtZ, dZtZInv, nz, dInvWork);
        CUDA_CHECK(cudaGetLastError());
        cudaFree(dInvWork);
        dInvWork = NULL;
    }

    CUDA_CHECK(cudaMalloc((void**)&dZtX, (size_t)nz * nx * sizeof(double)));
    if (causal_launch_matmul_t_dev(dZ, dX, dZtX, nz, T, nx) != 0) goto fail_iv;

    CUDA_CHECK(cudaMalloc((void**)&dProj, (size_t)nz * nx * sizeof(double)));
    if (causal_launch_matmul_dev(dZtZInv, dZtX, dProj, nz, nz, nx) != 0) goto fail_iv;

    CUDA_CHECK(cudaMalloc((void**)&dXHat, (size_t)T * nx * sizeof(double)));
    if (causal_launch_matmul_dev(dZ, dProj, dXHat, T, nz, nx) != 0) goto fail_iv;

    CUDA_CHECK(cudaMalloc((void**)&dXhTXh, (size_t)nx * nx * sizeof(double)));
    if (causal_launch_matmul_t_dev(dXHat, dXHat, dXhTXh, nx, T, nx) != 0) goto fail_iv;

    CUDA_CHECK(cudaMalloc((void**)&dXhTXhInv, (size_t)nx * nx * sizeof(double)));
    {
        size_t wz = ((size_t)nx * nx + (size_t)2 * nx) * sizeof(double);
        CUDA_CHECK(cudaMalloc((void**)&dInvWork, wz));
        invert_spd_global_kernel<<<1, 1>>>(dXhTXh, dXhTXhInv, nx, dInvWork);
        CUDA_CHECK(cudaGetLastError());
        cudaFree(dInvWork);
        dInvWork = NULL;
    }

    CUDA_CHECK(cudaMalloc((void**)&dXhTY, (size_t)nx * ny * sizeof(double)));
    if (causal_launch_matmul_t_dev(dXHat, dY, dXhTY, nx, T, ny) != 0) goto fail_iv;

    CUDA_CHECK(cudaMalloc((void**)&dBetaIv, (size_t)nx * ny * sizeof(double)));
    if (causal_launch_matmul_dev(dXhTXhInv, dXhTY, dBetaIv, nx, nx, ny) != 0) goto fail_iv;

    CUDA_CHECK(cudaMemcpy(
        beta_iv, dBetaIv, (size_t)nx * ny * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dZ); cudaFree(dX); cudaFree(dY);
    cudaFree(dZtZ); cudaFree(dZtZInv); cudaFree(dZtX); cudaFree(dProj);
    cudaFree(dXHat); cudaFree(dXhTXh); cudaFree(dXhTXhInv); cudaFree(dXhTY);
    cudaFree(dBetaIv);
    return 0;

fail_iv:
    if (dInvWork) cudaFree(dInvWork);
    cudaFree(dZ); cudaFree(dX); cudaFree(dY);
    if (dZtZ) cudaFree(dZtZ);
    if (dZtZInv) cudaFree(dZtZInv);
    if (dZtX) cudaFree(dZtX);
    if (dProj) cudaFree(dProj);
    if (dXHat) cudaFree(dXHat);
    if (dXhTXh) cudaFree(dXhTXh);
    if (dXhTXhInv) cudaFree(dXhTXhInv);
    if (dXhTY) cudaFree(dXhTY);
    if (dBetaIv) cudaFree(dBetaIv);
    return -1;
}

int cuda_causal_cate(
    const double* X, const double* treatment, const double* Y,
    double* cate,
    int T, int nx)
{
    double *dX = NULL, *dTr = NULL,      *dY = NULL;
    int *dIdx1 = NULL, *dIdx0 = NULL, *dC1 = NULL, *dC0 = NULL;
    double *dX1 = NULL, *dY1 = NULL, *dX0 = NULL, *dY0 = NULL;
    double *dBeta1 = NULL, *dBeta0 = NULL;
    double *dXTX = NULL, *dInv = NULL, *dXTY = NULL, *dIw = NULL;

    if (alloc_copy(X, (void**)&dX, (size_t)T * nx * sizeof(double))) return -1;
    if (alloc_copy(treatment, (void**)&dTr, (size_t)T * sizeof(double))) {
        cudaFree(dX); return -1;
    }
    if (alloc_copy(Y, (void**)&dY, (size_t)T * sizeof(double))) {
        cudaFree(dX); cudaFree(dTr); return -1;
    }

    CUDA_CHECK(cudaMalloc((void**)&dIdx1, (size_t)T * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dIdx0, (size_t)T * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dC1, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dC0, sizeof(int)));
    CUDA_CHECK(cudaMemset(dC1, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(dC0, 0, sizeof(int)));

    {
        int gridT = (T + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cate_fill_indices_kernel<<<gridT, BLOCK_SIZE>>>(dTr, T, dIdx1, dIdx0, dC1, dC0);
    }
    CUDA_CHECK(cudaGetLastError());

    int n1 = 0, n0 = 0;
    CUDA_CHECK(cudaMemcpy(&n1, dC1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&n0, dC0, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMalloc((void**)&dX1, (size_t)(n1 > 0 ? n1 : 1) * nx * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dY1, (size_t)(n1 > 0 ? n1 : 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dX0, (size_t)(n0 > 0 ? n0 : 1) * nx * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dY0, (size_t)(n0 > 0 ? n0 : 1) * sizeof(double)));

    if (n1 > 0) {
        int g1 = (n1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gather_X_rows_kernel<<<g1, BLOCK_SIZE>>>(dX, dIdx1, n1, nx, dX1);
        gather_y_values_kernel<<<g1, BLOCK_SIZE>>>(dY, dIdx1, n1, dY1);
        CUDA_CHECK(cudaGetLastError());
    }

    if (n0 > 0) {
        int g0 = (n0 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gather_X_rows_kernel<<<g0, BLOCK_SIZE>>>(dX, dIdx0, n0, nx, dX0);
        gather_y_values_kernel<<<g0, BLOCK_SIZE>>>(dY, dIdx0, n0, dY0);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMalloc((void**)&dBeta1, (size_t)nx * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dBeta0, (size_t)nx * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dXTX, (size_t)nx * nx * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dInv, (size_t)nx * nx * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dXTY, (size_t)nx * sizeof(double)));
    {
        size_t wz = ((size_t)nx * nx + (size_t)2 * nx) * sizeof(double);
        CUDA_CHECK(cudaMalloc((void**)&dIw, wz));
    }

    if (ols_fit_device_dim(dX1, n1, nx, dY1, dBeta1, dXTX, dInv, dXTY, dIw) != 0) {
        cudaFree(dX); cudaFree(dTr); cudaFree(dY);
        cudaFree(dIdx1); cudaFree(dIdx0); cudaFree(dC1); cudaFree(dC0);
        cudaFree(dX1); cudaFree(dY1); cudaFree(dX0); cudaFree(dY0);
        cudaFree(dBeta1); cudaFree(dBeta0); cudaFree(dXTX); cudaFree(dInv); cudaFree(dXTY);
        cudaFree(dIw);
        return -1;
    }

    if (ols_fit_device_dim(dX0, n0, nx, dY0, dBeta0, dXTX, dInv, dXTY, dIw) != 0) {
        cudaFree(dX); cudaFree(dTr); cudaFree(dY);
        cudaFree(dIdx1); cudaFree(dIdx0); cudaFree(dC1); cudaFree(dC0);
        cudaFree(dX1); cudaFree(dY1); cudaFree(dX0); cudaFree(dY0);
        cudaFree(dBeta1); cudaFree(dBeta0); cudaFree(dXTX); cudaFree(dInv); cudaFree(dXTY);
        cudaFree(dIw);
        return -1;
    }

    {
        double* dCate = NULL;
        CUDA_CHECK(cudaMalloc((void**)&dCate, (size_t)T * sizeof(double)));
        int gridT = (T + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cate_predict_kernel<<<gridT, BLOCK_SIZE>>>(dX, dBeta1, dBeta0, T, nx, dCate);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(cate, dCate, (size_t)T * sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(dCate);
    }

    cudaFree(dIw);

    cudaFree(dX); cudaFree(dTr); cudaFree(dY);
    cudaFree(dIdx1); cudaFree(dIdx0); cudaFree(dC1); cudaFree(dC0);
    cudaFree(dX1); cudaFree(dY1); cudaFree(dX0); cudaFree(dY0);
    cudaFree(dBeta1); cudaFree(dBeta0); cudaFree(dXTX); cudaFree(dInv); cudaFree(dXTY);

    return 0;
}

int cuda_causal_dag_markov(
    const double* X, const double* adj,
    double* log_prob,
    int T, int N)
{
    double *dX = NULL, *dLogProb = NULL, *dNodeVals = NULL;
    double *dMu = NULL, *dVar = NULL, *dRss = NULL;
    double *dPMat = NULL, *dBeta = NULL, *dXTX = NULL, *dInv = NULL, *dXTY = NULL;
    double *dIw = NULL;
    int *dParents = NULL;

    if (alloc_copy(X, (void**)&dX, (size_t)T * N * sizeof(double))) return -1;

    CUDA_CHECK(cudaMalloc((void**)&dLogProb, (size_t)T * sizeof(double)));
    CUDA_CHECK(cudaMemset(dLogProb, 0, (size_t)T * sizeof(double)));

    CUDA_CHECK(cudaMalloc((void**)&dNodeVals, (size_t)T * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dMu, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dVar, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dRss, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dParents, 64 * sizeof(int)));

    CUDA_CHECK(cudaMalloc((void**)&dPMat, (size_t)T * 64 * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dBeta, 64 * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dXTX, 64 * 64 * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dInv, 64 * 64 * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dXTY, 64 * sizeof(double)));
    {
        size_t wz = (64 * 64 + 2 * 64) * sizeof(double);
        CUDA_CHECK(cudaMalloc((void**)&dIw, wz));
    }

    int gridT = (T + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int node = 0; node < N; node++) {
        int parents[64], np = 0;

        for (int j = 0; j < N; j++) {
            if (adj[node * N + j] != 0.0) parents[np++] = j;
        }

        if (np > 64) {
            cudaFree(dIw); cudaFree(dX); cudaFree(dLogProb); cudaFree(dNodeVals);
            cudaFree(dMu); cudaFree(dVar); cudaFree(dRss); cudaFree(dParents);
            cudaFree(dPMat); cudaFree(dBeta); cudaFree(dXTX); cudaFree(dInv); cudaFree(dXTY);
            return -1;
        }

        CUDA_CHECK(cudaMemcpy(dParents, parents, (size_t)np * sizeof(int), cudaMemcpyHostToDevice));

        extract_column_kernel<<<gridT, BLOCK_SIZE>>>(dX, T, N, node, dNodeVals);
        CUDA_CHECK(cudaGetLastError());

        if (np == 0) {
            reduce_mean_var_kernel<<<1, 1>>>(dNodeVals, T, dMu, dVar);
            CUDA_CHECK(cudaGetLastError());

            double mu_h = 0.0, sigma2_h = 1e-10;
            CUDA_CHECK(cudaMemcpy(&mu_h, dMu, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&sigma2_h, dVar, sizeof(double), cudaMemcpyDeviceToHost));

            dag_logp_univariate_kernel<<<gridT, BLOCK_SIZE>>>(
                dNodeVals, mu_h, sigma2_h, dLogProb, T);
            CUDA_CHECK(cudaGetLastError());
        } else {
            dag_gather_parents_kernel<<<gridT, BLOCK_SIZE>>>(dX, T, N, dParents, np, dPMat);
            CUDA_CHECK(cudaGetLastError());

            if (ols_fit_device_dim(dPMat, T, np, dNodeVals, dBeta, dXTX, dInv, dXTY, dIw) != 0) {
                cudaFree(dIw); cudaFree(dX); cudaFree(dLogProb); cudaFree(dNodeVals);
                cudaFree(dMu); cudaFree(dVar); cudaFree(dRss); cudaFree(dParents);
                cudaFree(dPMat); cudaFree(dBeta); cudaFree(dXTX); cudaFree(dInv); cudaFree(dXTY);
                return -1;
            }

            rss_kernel<<<1, 1>>>(dPMat, dNodeVals, dBeta, T, np, dRss);
            CUDA_CHECK(cudaGetLastError());

            double rss_h = 0.0;
            CUDA_CHECK(cudaMemcpy(&rss_h, dRss, sizeof(double), cudaMemcpyDeviceToHost));

            double sigma2_h = fmax(rss_h / (double)T, 1e-10);

            dag_conditional_logp_kernel<<<gridT, BLOCK_SIZE>>>(
                dNodeVals, dX, T, N, dParents, np, dBeta, sigma2_h, dLogProb);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaMemcpy(log_prob, dLogProb, (size_t)T * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dIw);
    cudaFree(dX); cudaFree(dLogProb); cudaFree(dNodeVals);
    cudaFree(dMu); cudaFree(dVar); cudaFree(dRss); cudaFree(dParents);
    cudaFree(dPMat); cudaFree(dBeta); cudaFree(dXTX); cudaFree(dInv); cudaFree(dXTY);
    return 0;
}

int cuda_causal_counterfactual(
    const double* x_obs, const double* y_obs, const double* beta, const double* x_cf,
    double* out,
    int N, int N_cf)
{
    double *dX = NULL, *dY = NULL, *dBeta = NULL, *dXcf = NULL, *dOut = NULL;
    size_t nBytes = (size_t)N * sizeof(double);
    size_t cfBytes = (size_t)N_cf * sizeof(double);
    size_t outBytes = (size_t)N * N_cf * sizeof(double);

    if (alloc_copy(x_obs, (void**)&dX, nBytes)) return -1;
    if (alloc_copy(y_obs, (void**)&dY, nBytes)) goto fail;
    if (alloc_copy(beta, (void**)&dBeta, nBytes)) goto fail;
    if (alloc_copy(x_cf, (void**)&dXcf, cfBytes)) goto fail;
    if (cudaMalloc((void**)&dOut, outBytes) != cudaSuccess) goto fail;

    {
        int total = N * N_cf;
        counterfactual_kernel<<<(total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            dX, dY, dBeta, dXcf, dOut, N, N_cf
        );
    }

    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaMemcpy(out, dOut, outBytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(dX); cudaFree(dY); cudaFree(dBeta); cudaFree(dXcf); cudaFree(dOut);
    return 0;
fail:
    cudaFree(dX); cudaFree(dY); cudaFree(dBeta); cudaFree(dXcf); cudaFree(dOut);
    return -1;
}

int cuda_causal_frontdoor(
    const double* X, const double* M, const double* Y,
    double* effect,
    int T, int nx, int nm)
{
    double *dX = NULL, *dM = NULL, *dY = NULL;
    double *dXBoundaries = NULL, *dMBoundaries = NULL;
    double *dPX = NULL, *dCountX = NULL, *dMGivenX = NULL;
    double *dEYGivenXM = NULL, *dCountXM = NULL, *dEffect = NULL;
    int *dXBins = NULL, *dMBins = NULL;
    size_t sampleBytes = (size_t)T * sizeof(double);
    size_t xBoundaryBytes = (size_t)(nx - 1) * sizeof(double);
    size_t mBoundaryBytes = (size_t)(nm - 1) * sizeof(double);
    size_t xBytes = (size_t)nx * sizeof(double);
    size_t matrixBytes = (size_t)nx * nm * sizeof(double);
    int sampleBlocks = (T + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int matrixItems = nx * nm;
    int matrixBlocks = (matrixItems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (alloc_copy(X, (void**)&dX, sampleBytes)) return -1;
    if (alloc_copy(M, (void**)&dM, sampleBytes)) goto fail;
    if (alloc_copy(Y, (void**)&dY, sampleBytes)) goto fail;
    if (cudaMalloc((void**)&dXBins, (size_t)T * sizeof(int)) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&dMBins, (size_t)T * sizeof(int)) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&dPX, xBytes) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&dCountX, xBytes) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&dMGivenX, matrixBytes) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&dEYGivenXM, matrixBytes) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&dCountXM, matrixBytes) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&dEffect, xBytes) != cudaSuccess) goto fail;
    if (cudaMemset(dPX, 0, xBytes) != cudaSuccess) goto fail;
    if (cudaMemset(dCountX, 0, xBytes) != cudaSuccess) goto fail;
    if (cudaMemset(dMGivenX, 0, matrixBytes) != cudaSuccess) goto fail;
    if (cudaMemset(dEYGivenXM, 0, matrixBytes) != cudaSuccess) goto fail;
    if (cudaMemset(dCountXM, 0, matrixBytes) != cudaSuccess) goto fail;

    {
        thrust::device_ptr<double> xPtr(dX);
        thrust::device_ptr<double> mPtr(dM);
        thrust::device_vector<double> sortedX(xPtr, xPtr + T);
        thrust::device_vector<double> sortedM(mPtr, mPtr + T);
        thrust::sort(sortedX.begin(), sortedX.end());
        thrust::sort(sortedM.begin(), sortedM.end());

        if (cudaMalloc((void**)&dXBoundaries, xBoundaryBytes) != cudaSuccess) goto fail;
        if (cudaMalloc((void**)&dMBoundaries, mBoundaryBytes) != cudaSuccess) goto fail;

        fill_quantile_boundaries_kernel<<<(nx + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(sortedX.data()), dXBoundaries, T, nx
        );
        if (cudaGetLastError() != cudaSuccess) goto fail;
        fill_quantile_boundaries_kernel<<<(nm + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(sortedM.data()), dMBoundaries, T, nm
        );
        if (cudaGetLastError() != cudaSuccess) goto fail;
    }

    assign_frontdoor_bins_kernel<<<sampleBlocks, BLOCK_SIZE>>>(dX, dXBoundaries, dXBins, T, nx);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    assign_frontdoor_bins_kernel<<<sampleBlocks, BLOCK_SIZE>>>(dM, dMBoundaries, dMBins, T, nm);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    frontdoor_accumulate_kernel<<<sampleBlocks, BLOCK_SIZE>>>(
        dXBins, dMBins, dY, dPX, dCountX, dMGivenX, dEYGivenXM, dCountXM, T, nx, nm
    );
    if (cudaGetLastError() != cudaSuccess) goto fail;
    frontdoor_normalize_kernel<<<matrixBlocks, BLOCK_SIZE>>>(
        dPX, dCountX, dMGivenX, dEYGivenXM, dCountXM, T, nx, nm
    );
    if (cudaGetLastError() != cudaSuccess) goto fail;
    frontdoor_effect_kernel<<<(nx + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dPX, dMGivenX, dEYGivenXM, dEffect, nx, nm
    );
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaMemcpy(effect, dEffect, xBytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(dX); cudaFree(dM); cudaFree(dY);
    cudaFree(dXBoundaries); cudaFree(dMBoundaries);
    cudaFree(dPX); cudaFree(dCountX); cudaFree(dMGivenX);
    cudaFree(dEYGivenXM); cudaFree(dCountXM); cudaFree(dEffect);
    cudaFree(dXBins); cudaFree(dMBins);
    return 0;
fail:
    cudaFree(dX); cudaFree(dM); cudaFree(dY);
    cudaFree(dXBoundaries); cudaFree(dMBoundaries);
    cudaFree(dPX); cudaFree(dCountX); cudaFree(dMGivenX);
    cudaFree(dEYGivenXM); cudaFree(dCountXM); cudaFree(dEffect);
    cudaFree(dXBins); cudaFree(dMBins);
    return -1;
}

} // extern "C"
