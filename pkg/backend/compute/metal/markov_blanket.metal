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
    if (gid != 0) return;
    status[0] = 0;
    for (int idx = 0; idx < N; idx++) {
        float sf = masks[idx];
        float af = masks[N + idx];
        float inf = masks[2 * N + idx];
        float ef = masks[3 * N + idx];
        int c = (sf != 0.f) + (af != 0.f) + (inf != 0.f) + (ef != 0.f);
        if (c > 1) { status[0] = -2; return; }
    }
    int outLen = Ns + Na + Ni + Ne;
    for (int i = 0; i < outLen; i++) out[i] = 0.f;
    int si = 0, ai = Ns, ii = Ns + Na, ei = Ns + Na + Ni;
    for (int idx = 0; idx < N; idx++) {
        float v = x[idx];
        if (masks[idx] != 0.f && si < Ns) out[si++] = v;
        else if (masks[N + idx] != 0.f && ai < Ns + Na) out[ai++] = v;
        else if (masks[2 * N + idx] != 0.f && ii < Ns + Na + Ni) out[ii++] = v;
        else if (masks[3 * N + idx] != 0.f && ei < outLen) out[ei++] = v;
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
    if (gid >= (uint)Ni) return;
    float s = bias[gid];
    for (int j = 0; j < Ns; j++) s += W[gid * Ns + j] * x_sens[j];
    out[gid] = s;
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
    if (gid >= (uint)Na) return;
    float s = bias[gid];
    for (int j = 0; j < Ni; j++) s += W[gid * Ni + j] * x_int[j];
    out[gid] = s;
}

// --- Mutual Information Parallel Kernels ---

kernel void mb_mean_kernel(
    device const float* data [[buffer(0)]],
    device float* mean [[buffer(1)]],
    constant int& T [[buffer(2)]],
    constant int& D [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)D) return;
    float s = 0.f;
    for (int t = 0; t < T; t++) s += data[t * D + gid];
    mean[gid] = s / (float)T;
}

kernel void mb_cov_kernel(
    device const float* X [[buffer(0)]],
    device const float* Y [[buffer(1)]],
    device const float* xm [[buffer(2)]],
    device const float* ym [[buffer(3)]],
    device float* cov [[buffer(4)]],
    constant int& T [[buffer(5)]],
    constant int& D1 [[buffer(6)]],
    constant int& D2 [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.x;
    uint c = gid.y;
    if (r >= (uint)D1 || c >= (uint)D2) return;
    float s = 0.f;
    float mr = xm[r];
    float mc = ym[c];
    for (int t = 0; t < T; t++) {
        s += (X[t * D1 + r] - mr) * (Y[t * D2 + c] - mc);
    }
    cov[r * D2 + c] = s / (float)(T - 1);
}

kernel void mb_joint_kernel(
    device const float* cx [[buffer(0)]],
    device const float* cy [[buffer(1)]],
    device const float* cxy [[buffer(2)]],
    device float* joint [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& M [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint r = gid.x;
    uint c = gid.y;
    int NM = N + M;
    if (r >= (uint)NM || c >= (uint)NM) return;

    if (r < (uint)N && c < (uint)N) {
        joint[r * NM + c] = cx[r * N + c];
    } else if (r >= (uint)N && c >= (uint)N) {
        joint[r * NM + c] = cy[(r - N) * M + (c - N)];
    } else if (r < (uint)N && c >= (uint)N) {
        joint[r * NM + c] = cxy[r * M + (c - N)];
    } else {
        joint[r * NM + c] = cxy[c * M + (r - N)];
    }
}

float mb_logdet_cholesky(device const float* src, int dim, device float* work)
{
    for (int i = 0; i < dim * dim; i++) work[i] = src[i];
    const float eps = 1e-6f;
    for (int d = 0; d < dim; d++) work[d * dim + d] += eps;

    for (int j = 0; j < dim; j++) {
        float sum = work[j * dim + j];
        for (int k = 0; k < j; k++) {
            float v = work[j * dim + k];
            sum -= v * v;
        }
        if (sum <= 0.f) return NAN;
        float Ljj = sqrt(sum);
        work[j * dim + j] = Ljj;
        float inv = 1.f / Ljj;
        for (int i = j + 1; i < dim; i++) {
            float s = work[i * dim + j];
            for (int k = 0; k < j; k++) s -= work[i * dim + k] * work[j * dim + k];
            work[i * dim + j] = s * inv;
        }
    }
    float logdet = 0.f;
    for (int d = 0; d < dim; d++) logdet += log(work[d * dim + d]);
    return 2.f * logdet;
}

kernel void mb_chol_logdet_kernel(
    device const float* cx [[buffer(0)]],
    device const float* cy [[buffer(1)]],
    device const float* joint [[buffer(2)]],
    device float* mi_out [[buffer(3)]],
    device float* work [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& M [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    int NM = N + M;
    float ldX = mb_logdet_cholesky(cx, N, work);
    float ldY = mb_logdet_cholesky(cy, M, work);
    float ldJ = mb_logdet_cholesky(joint, NM, work);

    if (isnan(ldX) || isnan(ldY) || isnan(ldJ)) {
        mi_out[0] = NAN;
        return;
    }
    float mi = 0.5f * (ldX + ldY - ldJ);
    if (mi < 0.f || isnan(mi) || isinf(mi)) mi = 0.f;
    mi_out[0] = mi;
}
