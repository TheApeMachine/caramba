#include <metal_stdlib>
using namespace metal;

// --- Parallel LA Kernels ---

kernel void causal_axpy_kernel(
    device       float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    constant     float& scale [[buffer(2)]],
    constant     int& n [[buffer(3)]],
    uint         gid [[thread_position_in_grid]])
{
    if (gid >= (uint)n) return;
    dst[gid] += scale * src[gid];
}

kernel void causal_sub_kernel(
    device       float* dst [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],
    constant     int& n [[buffer(3)]],
    uint         gid [[thread_position_in_grid]])
{
    if (gid >= (uint)n) return;
    dst[gid] = a[gid] - b[gid];
}

kernel void causal_matvec_kernel(
    device       float* dst [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* x [[buffer(2)]],
    constant     int& rows [[buffer(3)]],
    constant     int& cols [[buffer(4)]],
    uint         gid [[thread_position_in_grid]])
{
    if (gid >= (uint)rows) return;
    float s = 0.f;
    for (int j = 0; j < cols; j++) {
        s += W[gid * cols + j] * x[j];
    }
    dst[gid] = s;
}

kernel void causal_dot_atomic_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device atomic_float* sum [[buffer(2)]],
    constant     int& n [[buffer(3)]],
    uint         gid [[thread_position_in_grid]])
{
    if (gid >= (uint)n) return;
    atomic_fetch_add_explicit(sum, a[gid] * b[gid], memory_order_relaxed);
}

kernel void causal_ata_kernel(
    device const float* A [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int& T [[buffer(2)]],
    constant int& p [[buffer(3)]],
    constant float& ridge [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.x;
    uint c = gid.y;
    if (r >= (uint)p || c >= (uint)p) return;
    float s = 0.f;
    for (int t = 0; t < T; t++) {
        s += A[t * p + r] * A[t * p + c];
    }
    if (r == c) s += ridge;
    out[r * p + c] = s;
}

kernel void causal_atb_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& T [[buffer(3)]],
    constant int& pA [[buffer(4)]],
    constant int& pB [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.x;
    uint c = gid.y;
    if (r >= (uint)pA || c >= (uint)pB) return;
    float s = 0.f;
    for (int t = 0; t < T; t++) {
        s += A[t * pA + r] * B[t * pB + c];
    }
    out[r * pB + c] = s;
}

kernel void causal_matmul_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.x;
    uint c = gid.y;
    if (r >= (uint)M || c >= (uint)N) return;
    float s = 0.f;
    for (int k = 0; k < K; k++) {
        s += A[r * K + k] * B[k * N + c];
    }
    out[r * N + c] = s;
}

// Single-thread Cholesky inversion for small matrices
kernel void causal_chol_inv_kernel(
    device const float* src [[buffer(0)]],
    device float* inv [[buffer(1)]],
    device float* work [[buffer(2)]],
    device int* err [[buffer(3)]],
    constant int& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    err[0] = 0;
    device float* a = work;
    device float* yCol = work + p * p;
    device float* xCol = work + p * p + p;
    for (int i = 0; i < p * p; i++) a[i] = src[i];

    for (int row = 0; row < p; row++) {
        for (int col = 0; col <= row; col++) {
            float sum = a[row * p + col];
            for (int k = 0; k < col; k++) {
                sum -= a[row * p + k] * a[col * p + k];
            }
            if (row == col) {
                if (sum <= 0.f) sum = 1e-10f;
                a[row * p + col] = sqrt(sum);
            } else {
                a[row * p + col] = sum / a[col * p + col];
            }
        }
        for (int col = row + 1; col < p; col++) {
            a[row * p + col] = 0.f;
        }
    }

    for (int col = 0; col < p; col++) {
        for (int row = 0; row < p; row++) yCol[row] = 0.f;
        yCol[col] = 1.f;
        for (int row = col; row < p; row++) {
            if (row > col) {
                yCol[row] = 0.f;
                for (int k = col; k < row; k++) {
                    yCol[row] -= a[row * p + k] * yCol[k];
                }
            }
            yCol[row] /= a[row * p + row];
        }
        for (int row = p - 1; row >= 0; row--) {
            xCol[row] = yCol[row];
            for (int k = row + 1; k < p; k++) {
                xCol[row] -= a[k * p + row] * xCol[k];
            }
            xCol[row] /= a[row * p + row];
        }
        for (int row = 0; row < p; row++) {
            inv[row * p + col] = xCol[row];
        }
    }
}

// --- Do-Calculus Kernels ---

kernel void docalc_extract_kernel(
    device const float* cov [[buffer(0)]],
    device const float* mask [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* sigII [[buffer(3)]],
    device float* sigFI [[buffer(4)]],
    device float* sigFF [[buffer(5)]],
    device float* sigIF [[buffer(6)]],
    device float* xIntV [[buffer(7)]],
    device int* intervened [[buffer(8)]],
    device int* freev [[buffer(9)]],
    device int* counts [[buffer(10)]],
    constant int& N [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    int ni = 0, nf = 0;
    for (int i = 0; i < N; i++) {
        if (mask[i] != 0.f) intervened[ni++] = i;
        else freev[nf++] = i;
    }
    counts[0] = ni;
    counts[1] = nf;

    for (int r = 0; r < ni; r++) {
        xIntV[r] = values[intervened[r]];
        for (int c = 0; c < ni; c++) sigII[r * ni + c] = cov[intervened[r] * N + intervened[c]];
        for (int c = 0; c < nf; c++) sigIF[r * nf + c] = cov[intervened[r] * N + freev[c]];
    }
    for (int r = 0; r < nf; r++) {
        for (int c = 0; c < ni; c++) sigFI[r * ni + c] = cov[freev[r] * N + intervened[c]];
        for (int c = 0; c < nf; c++) sigFF[r * nf + c] = cov[freev[r] * N + freev[c]];
    }
}

kernel void docalc_assemble_kernel(
    device float* out [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const float* delta [[buffer(2)]],
    device const float* sigFF [[buffer(3)]],
    device const float* correction [[buffer(4)]],
    device const int* intervened [[buffer(5)]],
    device const int* freev [[buffer(6)]],
    device const int* counts [[buffer(7)]],
    constant int& N [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    int ni = counts[0];
    int nf = counts[1];
    
    device float* adjMean = out;
    device float* adjCov = out + N;
    
    for (int i = 0; i < N * N; i++) adjCov[i] = 0.f;
    for (int i = 0; i < N; i++) adjMean[i] = 0.f;

    for (int k = 0; k < ni; k++) {
        adjMean[intervened[k]] = values[intervened[k]];
    }
    for (int k = 0; k < nf; k++) {
        adjMean[freev[k]] = delta[k];
        for (int c = 0; c < nf; c++) {
            adjCov[freev[k] * N + freev[c]] = sigFF[k * nf + c] - correction[k * nf + c];
        }
    }
}

// --- Backdoor Kernels ---

kernel void backdoor_design_kernel(
    device const float* X [[buffer(0)]],
    device const float* Z [[buffer(1)]],
    device float* design [[buffer(2)]],
    constant int& T [[buffer(3)]],
    constant int& nx [[buffer(4)]],
    constant int& nz [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint t = gid.x;
    uint c = gid.y;
    int p = 1 + nx + nz;
    if (t >= (uint)T || c >= (uint)p) return;

    if (c == 0) {
        design[t * p] = 1.f;
    } else if (c < (uint)(1 + nx)) {
        design[t * p + c] = X[t * nx + (c - 1)];
    } else {
        design[t * p + c] = Z[t * nz + (c - 1 - nx)];
    }
}

kernel void backdoor_effect_kernel(
    device const float* beta [[buffer(0)]],
    device float* effect [[buffer(1)]],
    constant int& ny [[buffer(2)]],
    constant int& nx [[buffer(3)]],
    constant int& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)ny) return;
    float eff = 0.f;
    for (int xDim = 0; xDim < nx; xDim++) {
        float b = beta[gid * p + 1 + xDim];
        eff += (b >= 0.f) ? b : -b;
    }
    effect[gid] = eff / (float)nx;
}

// --- CATE Kernels ---

kernel void cate_split_kernel(
    device const float* X [[buffer(0)]],
    device const float* treatment [[buffer(1)]],
    device const float* Y [[buffer(2)]],
    device float* xSub1 [[buffer(3)]],
    device float* ySub1 [[buffer(4)]],
    device float* xSub0 [[buffer(5)]],
    device float* ySub0 [[buffer(6)]],
    device int* counts [[buffer(7)]],
    constant int& T [[buffer(8)]],
    constant int& nx [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    int nt = 0, nc = 0;
    int nFeat = nx + 1;
    for (int i = 0; i < T; i++) {
        if (treatment[i] >= 0.5f) {
            xSub1[nt * nFeat] = 1.f;
            for (int j = 0; j < nx; j++) xSub1[nt * nFeat + 1 + j] = X[i * nx + j];
            ySub1[nt] = Y[i];
            nt++;
        } else {
            xSub0[nc * nFeat] = 1.f;
            for (int j = 0; j < nx; j++) xSub0[nc * nFeat + 1 + j] = X[i * nx + j];
            ySub0[nc] = Y[i];
            nc++;
        }
    }
    counts[0] = nt;
    counts[1] = nc;
}

kernel void cate_effect_kernel(
    device const float* X [[buffer(0)]],
    device const float* b1 [[buffer(1)]],
    device const float* b0 [[buffer(2)]],
    device float* cate [[buffer(3)]],
    constant int& T [[buffer(4)]],
    constant int& nx [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)T) return;
    float p1 = b1[0];
    float p0 = b0[0];
    for (int j = 0; j < nx; j++) {
        float xv = X[gid * nx + j];
        p1 += b1[1 + j] * xv;
        p0 += b0[1 + j] * xv;
    }
    cate[gid] = p1 - p0;
}

// --- DAG Markov Kernels ---

kernel void dag_markov_prep_kernel(
    device const float* X [[buffer(0)]],
    device const float* adj [[buffer(1)]],
    device float* pMat [[buffer(2)]],
    device float* nodeVals [[buffer(3)]],
    device int* counts [[buffer(4)]],
    constant int& T [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& nodeIdx [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    int np = 0;
    for (int j = 0; j < N; j++) {
        if (adj[nodeIdx * N + j] != 0.f) {
            for (int o = 0; o < T; o++) {
                pMat[o * (N + 1) + 1 + np] = X[o * N + j];
            }
            np++;
        }
    }
    for (int o = 0; o < T; o++) {
        pMat[o * (N + 1)] = 1.f;
        nodeVals[o] = X[o * N + nodeIdx];
    }
    counts[0] = np;
}

kernel void dag_markov_score_kernel(
    device const float* X [[buffer(0)]],
    device const float* adj [[buffer(1)]],
    device const float* betas [[buffer(2)]],
    device const float* sigma2 [[buffer(3)]],
    device float* log_prob [[buffer(4)]],
    constant int& T [[buffer(5)]],
    constant int& N [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)T) return;
    float logP = 0.f;
    const float pi = 3.14159265f;
    for (int nodeIdx = 0; nodeIdx < N; nodeIdx++) {
        float s2 = sigma2[nodeIdx];
        float xVal = X[gid * N + nodeIdx];
        device const float* b = betas + nodeIdx * (N + 1);
        float pred = b[0];
        int pk = 0;
        for (int j = 0; j < N; j++) {
            if (adj[nodeIdx * N + j] != 0.f) {
                pred += b[1 + pk] * X[gid * N + j];
                pk++;
            }
        }
        float diff = xVal - pred;
        logP += -0.5f * log(2.f * pi * s2) - 0.5f * diff * diff / s2;
    }
    log_prob[gid] = logP;
}
