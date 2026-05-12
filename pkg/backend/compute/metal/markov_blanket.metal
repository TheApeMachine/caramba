#include <metal_stdlib>
using namespace metal;

kernel void mb_partition_kernel(
    device const float* x [[buffer(0)]],
    device const float* masks [[buffer(1)]],
    device       float* out [[buffer(2)]],
    device       int* status [[buffer(3)]],
    constant     int& N [[buffer(4)]],
    constant     int& Ns [[buffer(5)]],
    constant     int& Na [[buffer(6)]],
    constant     int& Ni [[buffer(7)]],
    constant     int& Ne [[buffer(8)]],
    uint         gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    status[0] = 0;

    for (int idx = 0; idx < N; idx++) {
        float sf = masks[idx];
        float af = masks[N + idx];
        float inf = masks[2 * N + idx];
        float ef = masks[3 * N + idx];
        int c = (sf != 0.f) + (af != 0.f) + (inf != 0.f) + (ef != 0.f);
        if (c > 1) {
            status[0] = -2;
            return;
        }
    }

    int outLen = Ns + Na + Ni + Ne;
    for (int i = 0; i < outLen; i++) {
        out[i] = 0.f;
    }

    int si = 0;
    int ai = Ns;
    int ii = Ns + Na;
    int ei = Ns + Na + Ni;

    for (int idx = 0; idx < N; idx++) {
        float v = x[idx];
        if (masks[idx] != 0.f && si < Ns) {
            out[si++] = v;
        } else if (masks[N + idx] != 0.f && ai < Ns + Na) {
            out[ai++] = v;
        } else if (masks[2 * N + idx] != 0.f && ii < Ns + Na + Ni) {
            out[ii++] = v;
        } else if (masks[3 * N + idx] != 0.f && ei < outLen) {
            out[ei++] = v;
        }
    }
}

kernel void mb_flow_internal_kernel(
    device const float* x_sens [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device       float* out [[buffer(3)]],
    constant     int& Ni [[buffer(4)]],
    constant     int& Ns [[buffer(5)]],
    uint         gid [[thread_position_in_grid]])
{
    if (gid >= (uint)Ni) {
        return;
    }
    uint row = gid;
    float s = bias[row];
    for (int j = 0; j < Ns; j++) {
        s += W[row * Ns + j] * x_sens[j];
    }
    out[row] = s;
}

kernel void mb_flow_active_kernel(
    device const float* x_int [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device       float* out [[buffer(3)]],
    constant     int& Na [[buffer(4)]],
    constant     int& Ni [[buffer(5)]],
    uint         gid [[thread_position_in_grid]])
{
    if (gid >= (uint)Na) {
        return;
    }
    uint row = gid;
    float s = bias[row];
    for (int j = 0; j < Ni; j++) {
        s += W[row * Ni + j] * x_int[j];
    }
    out[row] = s;
}

// log|Sigma| for SPD copy (row-major lower Cholesky); returns nan if fails.
float mb_logdet_cholesky(device const float* src, int dim, device float* work)
{
    for (int i = 0; i < dim * dim; i++) {
        work[i] = src[i];
    }
    const float eps = 1e-6f;
    for (int d = 0; d < dim; d++) {
        work[d * dim + d] += eps;
    }

    for (int j = 0; j < dim; j++) {
        float sum = work[j * dim + j];
        for (int k = 0; k < j; k++) {
            float v = work[j * dim + k];
            sum -= v * v;
        }
        if (sum <= 0.f) {
            return NAN;
        }
        float Ljj = sqrt(sum);
        work[j * dim + j] = Ljj;
        float inv = 1.f / Ljj;
        for (int i = j + 1; i < dim; i++) {
            float s = work[i * dim + j];
            for (int k = 0; k < j; k++) {
                s -= work[i * dim + k] * work[j * dim + k];
            }
            work[i * dim + j] = s * inv;
        }
    }

    float logdet = 0.f;
    for (int d = 0; d < dim; d++) {
        logdet += log(work[d * dim + d]);
    }
    return 2.f * logdet;
}

kernel void mb_mutual_information_kernel(
    device const float* X [[buffer(0)]],
    device const float* Y [[buffer(1)]],
    device       float* mi_out [[buffer(2)]],
    device       float* scratch [[buffer(3)]],
    constant     int& T [[buffer(4)]],
    constant     int& N [[buffer(5)]],
    constant     int& M [[buffer(6)]],
    uint         gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }

    if (T < 2 || N <= 0 || M <= 0) {
        mi_out[0] = NAN;
        return;
    }

    int NM = N + M;
    int off_mean = 0;
    int off_xm = off_mean;
    int off_ym = off_xm + N;
    int off_cx = off_ym + M;
    int off_cy = off_cx + N * N;
    int off_cxy = off_cy + M * M;
    int off_joint = off_cxy + N * M;
    int off_chol = off_joint + NM * NM;

    device float* xm = scratch + off_xm;
    device float* ym = scratch + off_ym;
    device float* cx = scratch + off_cx;
    device float* cy = scratch + off_cy;
    device float* cxy = scratch + off_cxy;
    device float* joint = scratch + off_joint;
    device float* cholwork = scratch + off_chol;

    for (int d = 0; d < N; d++) {
        xm[d] = 0.f;
    }
    for (int d = 0; d < M; d++) {
        ym[d] = 0.f;
    }

    for (int t = 0; t < T; t++) {
        for (int d = 0; d < N; d++) {
            xm[d] += X[t * N + d];
        }
        for (int d = 0; d < M; d++) {
            ym[d] += Y[t * M + d];
        }
    }
    float invT = 1.f / (float)T;
    for (int d = 0; d < N; d++) {
        xm[d] *= invT;
    }
    for (int d = 0; d < M; d++) {
        ym[d] *= invT;
    }

    for (int i = 0; i < N * N; i++) {
        cx[i] = 0.f;
    }
    for (int i = 0; i < M * M; i++) {
        cy[i] = 0.f;
    }
    for (int i = 0; i < N * M; i++) {
        cxy[i] = 0.f;
    }

    if (T <= 1) {
        mi_out[0] = 0.f;
        return;
    }

    float invTm1 = 1.f / (float)(T - 1);

    for (int t = 0; t < T; t++) {
        for (int row = 0; row < N; row++) {
            float dr = X[t * N + row] - xm[row];
            for (int col = 0; col < N; col++) {
                float dc = X[t * N + col] - xm[col];
                cx[row * N + col] += dr * dc;
            }
        }
        for (int row = 0; row < M; row++) {
            float dr = Y[t * M + row] - ym[row];
            for (int col = 0; col < M; col++) {
                float dc = Y[t * M + col] - ym[col];
                cy[row * M + col] += dr * dc;
            }
        }
        for (int row = 0; row < N; row++) {
            float dr = X[t * N + row] - xm[row];
            for (int col = 0; col < M; col++) {
                cxy[row * M + col] += dr * (Y[t * M + col] - ym[col]);
            }
        }
    }

    for (int i = 0; i < N * N; i++) {
        cx[i] *= invTm1;
    }
    for (int i = 0; i < M * M; i++) {
        cy[i] *= invTm1;
    }
    for (int i = 0; i < N * M; i++) {
        cxy[i] *= invTm1;
    }

    for (int i = 0; i < NM * NM; i++) {
        joint[i] = 0.f;
    }
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            joint[row * NM + col] = cx[row * N + col];
        }
    }
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < M; col++) {
            joint[(N + row) * NM + (N + col)] = cy[row * M + col];
        }
    }
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < M; col++) {
            float v = cxy[row * M + col];
            joint[row * NM + (N + col)] = v;
            joint[(N + col) * NM + row] = v;
        }
    }

    float ldX = mb_logdet_cholesky(cx, N, cholwork);
    float ldY = mb_logdet_cholesky(cy, M, cholwork);
    float ldJ = mb_logdet_cholesky(joint, NM, cholwork);

    if (isnan(ldX) || isnan(ldY) || isnan(ldJ)) {
        mi_out[0] = NAN;
        return;
    }

    float mi = 0.5f * (ldX + ldY - ldJ);
    if (mi < 0.f || isnan(mi) || isinf(mi)) {
        mi = 0.f;
    }
    mi_out[0] = mi;
}
