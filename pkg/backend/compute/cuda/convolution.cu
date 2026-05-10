#include <cuda_runtime.h>
#include "convolution.h"

// atomicAdd for double requires sm_60+
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* addr, double val) {
    unsigned long long int* addr_ull = (unsigned long long int*)addr;
    unsigned long long int old = *addr_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// ---------------------------------------------------------------------------
// Conv1d kernel — one thread per output element [n, oc, l_out]
// ---------------------------------------------------------------------------

__global__ void conv1d_kernel(
    const double* x, double* dst,
    int N, int InC, int L,
    int OutC, int K,
    int stride, int pad, int dilation, int groups, int L_out,
    const double* weight, const double* bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OutC * L_out;
    if (idx >= total) return;

    int lo  = idx % L_out;
    int tmp = idx / L_out;
    int oc  = tmp % OutC;
    int ni  = tmp / OutC;

    int ocPerGroup = OutC / groups;
    int icPerGroup = InC  / groups;
    int g          = oc / ocPerGroup;
    int ocLocal    = oc % ocPerGroup;
    int icStart    = g * icPerGroup;
    int kernElems  = icPerGroup * K;
    const double* wRow = weight + (g * ocPerGroup + ocLocal) * kernElems;

    double sum = bias[oc];
    for (int ic = 0; ic < icPerGroup; ic++) {
        int absIC = icStart + ic;
        for (int k = 0; k < K; k++) {
            int li = lo * stride + k * dilation - pad;
            if (li >= 0 && li < L) {
                sum += x[ni * InC * L + absIC * L + li] * wRow[ic * K + k];
            }
        }
    }
    dst[ni * OutC * L_out + oc * L_out + lo] = sum;
}

// ---------------------------------------------------------------------------
// Conv2d kernel — one thread per output element [n, oc, h_out, w_out]
// ---------------------------------------------------------------------------

__global__ void conv2d_kernel(
    const double* x, double* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const double* weight, const double* bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OutC * Hout * Wout;
    if (idx >= total) return;

    int wo  = idx % Wout;
    int tmp = idx / Wout;
    int ho  = tmp % Hout;
    tmp     = tmp / Hout;
    int oc  = tmp % OutC;
    int ni  = tmp / OutC;

    int ocPerGroup = OutC / groups;
    int icPerGroup = InC  / groups;
    int g          = oc / ocPerGroup;
    int ocLocal    = oc % ocPerGroup;
    int icStart    = g * icPerGroup;
    int kernElems  = icPerGroup * KH * KW;
    const double* wRow = weight + (g * ocPerGroup + ocLocal) * kernElems;

    double sum = bias[oc];
    for (int ic = 0; ic < icPerGroup; ic++) {
        int absIC = icStart + ic;
        for (int kh = 0; kh < KH; kh++) {
            int hi = ho * sH + kh * dH - pH;
            if (hi < 0 || hi >= H) continue;
            for (int kw = 0; kw < KW; kw++) {
                int wi = wo * sW + kw * dW - pW;
                if (wi >= 0 && wi < W) {
                    sum += x[ni*InC*H*W + absIC*H*W + hi*W + wi]
                         * wRow[ic*KH*KW + kh*KW + kw];
                }
            }
        }
    }
    dst[ni*OutC*Hout*Wout + oc*Hout*Wout + ho*Wout + wo] = sum;
}

// ---------------------------------------------------------------------------
// Conv3d kernel — one thread per output element [n, oc, d_out, h_out, w_out]
// ---------------------------------------------------------------------------

__global__ void conv3d_kernel(
    const double* x, double* dst,
    int N, int InC, int D, int H, int W,
    int OutC, int KD, int KH, int KW,
    int sD, int sH, int sW, int pD, int pH, int pW,
    int dD, int dH, int dW, int groups,
    int Dout, int Hout, int Wout,
    const double* weight, const double* bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OutC * Dout * Hout * Wout;
    if (idx >= total) return;

    int wo  = idx % Wout;
    int tmp = idx / Wout;
    int ho  = tmp % Hout;
    tmp     = tmp / Hout;
    int doo = tmp % Dout;
    tmp     = tmp / Dout;
    int oc  = tmp % OutC;
    int ni  = tmp / OutC;

    int ocPerGroup = OutC / groups;
    int icPerGroup = InC  / groups;
    int g          = oc / ocPerGroup;
    int ocLocal    = oc % ocPerGroup;
    int icStart    = g * icPerGroup;
    int kernElems  = icPerGroup * KD * KH * KW;
    const double* wRow = weight + (g * ocPerGroup + ocLocal) * kernElems;

    double sum = bias[oc];
    for (int ic = 0; ic < icPerGroup; ic++) {
        int absIC = icStart + ic;
        for (int kd = 0; kd < KD; kd++) {
            int di = doo * sD + kd * dD - pD;
            if (di < 0 || di >= D) continue;
            for (int kh = 0; kh < KH; kh++) {
                int hi = ho * sH + kh * dH - pH;
                if (hi < 0 || hi >= H) continue;
                for (int kw = 0; kw < KW; kw++) {
                    int wi = wo * sW + kw * dW - pW;
                    if (wi >= 0 && wi < W) {
                        int xIdx = ni*InC*D*H*W + absIC*D*H*W + di*H*W + hi*W + wi;
                        int wIdx = ic*KD*KH*KW + kd*KH*KW + kh*KW + kw;
                        sum += x[xIdx] * wRow[wIdx];
                    }
                }
            }
        }
    }
    dst[ni*OutC*Dout*Hout*Wout + oc*Dout*Hout*Wout + doo*Hout*Wout + ho*Wout + wo] = sum;
}

// ---------------------------------------------------------------------------
// ConvTranspose2d kernel — one thread per input pixel [n, ic, h, w]
// Scatter-adds to output using atomicAdd.
// ---------------------------------------------------------------------------

__global__ void conv_transpose2d_kernel(
    const double* x, double* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const double* weight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * InC * H * W;
    if (idx >= total) return;

    int wi  = idx % W;
    int tmp = idx / W;
    int hi  = tmp % H;
    tmp     = tmp / H;
    int ic  = tmp % InC;
    int ni  = tmp / InC;

    int ocPerGroup = OutC / groups;
    int icPerGroup = InC  / groups;
    int g          = ic / icPerGroup;
    int ocStart    = g * ocPerGroup;
    int kernElems  = ocPerGroup * KH * KW;
    const double* wRow = weight + ic * kernElems;

    double xVal = x[ni*InC*H*W + ic*H*W + hi*W + wi];

    for (int oc = 0; oc < ocPerGroup; oc++) {
        int absOC = ocStart + oc;
        for (int kh = 0; kh < KH; kh++) {
            int ho = hi * sH + kh * dH - pH;
            if (ho < 0 || ho >= Hout) continue;
            for (int kw = 0; kw < KW; kw++) {
                int wo = wi * sW + kw * dW - pW;
                if (wo >= 0 && wo < Wout) {
                    int oIdx = ni*OutC*Hout*Wout + absOC*Hout*Wout + ho*Wout + wo;
                    atomicAdd(&dst[oIdx], xVal * wRow[oc*KH*KW + kh*KW + kw]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

static const int BLOCK = 256;
static inline int blocks(int n) { return (n + BLOCK - 1) / BLOCK; }

// ---------------------------------------------------------------------------
// C linkage wrappers
// ---------------------------------------------------------------------------

extern "C" {

int cuda_conv1d(
    const double* x, double* dst,
    int N, int InC, int L,
    int OutC, int K,
    int stride, int pad, int dilation, int groups, int L_out,
    const double* weight, const double* bias)
{
    int xn = N * InC * L;
    int dn = N * OutC * L_out;
    int wn = OutC * (InC / groups) * K;
    int bn = OutC;

    double *d_x = NULL, *d_dst = NULL, *d_w = NULL, *d_b = NULL;
    if (cudaMalloc(&d_x,   xn * sizeof(double)) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, dn * sizeof(double)) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_w,   wn * sizeof(double)) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_b,   bn * sizeof(double)) != cudaSuccess) goto fail;

    if (cudaMemcpy(d_x, x,      xn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_w, weight, wn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_b, bias,   bn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    conv1d_kernel<<<blocks(dn), BLOCK>>>(
        d_x, d_dst, N, InC, L, OutC, K,
        stride, pad, dilation, groups, L_out, d_w, d_b);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, dn * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_x); cudaFree(d_dst); cudaFree(d_w); cudaFree(d_b);
    return 0;
fail:
    cudaFree(d_x); cudaFree(d_dst); cudaFree(d_w); cudaFree(d_b);
    return -1;
}

int cuda_conv2d(
    const double* x, double* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const double* weight, const double* bias)
{
    int xn = N * InC * H * W;
    int dn = N * OutC * Hout * Wout;
    int wn = OutC * (InC / groups) * KH * KW;
    int bn = OutC;

    double *d_x = NULL, *d_dst = NULL, *d_w = NULL, *d_b = NULL;
    if (cudaMalloc(&d_x,   xn * sizeof(double)) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, dn * sizeof(double)) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_w,   wn * sizeof(double)) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_b,   bn * sizeof(double)) != cudaSuccess) goto fail;

    if (cudaMemcpy(d_x, x,      xn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_w, weight, wn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_b, bias,   bn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    conv2d_kernel<<<blocks(dn), BLOCK>>>(
        d_x, d_dst, N, InC, H, W, OutC, KH, KW,
        sH, sW, pH, pW, dH, dW, groups, Hout, Wout, d_w, d_b);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, dn * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_x); cudaFree(d_dst); cudaFree(d_w); cudaFree(d_b);
    return 0;
fail:
    cudaFree(d_x); cudaFree(d_dst); cudaFree(d_w); cudaFree(d_b);
    return -1;
}

int cuda_conv3d(
    const double* x, double* dst,
    int N, int InC, int D, int H, int W,
    int OutC, int KD, int KH, int KW,
    int sD, int sH, int sW, int pD, int pH, int pW,
    int dD, int dH, int dW, int groups,
    int Dout, int Hout, int Wout,
    const double* weight, const double* bias)
{
    int xn = N * InC * D * H * W;
    int dn = N * OutC * Dout * Hout * Wout;
    int wn = OutC * (InC / groups) * KD * KH * KW;
    int bn = OutC;

    double *d_x = NULL, *d_dst = NULL, *d_w = NULL, *d_b = NULL;
    if (cudaMalloc(&d_x,   xn * sizeof(double)) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, dn * sizeof(double)) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_w,   wn * sizeof(double)) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_b,   bn * sizeof(double)) != cudaSuccess) goto fail;

    if (cudaMemcpy(d_x, x,      xn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_w, weight, wn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_b, bias,   bn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    conv3d_kernel<<<blocks(dn), BLOCK>>>(
        d_x, d_dst, N, InC, D, H, W, OutC, KD, KH, KW,
        sD, sH, sW, pD, pH, pW, dD, dH, dW, groups,
        Dout, Hout, Wout, d_w, d_b);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, dn * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_x); cudaFree(d_dst); cudaFree(d_w); cudaFree(d_b);
    return 0;
fail:
    cudaFree(d_x); cudaFree(d_dst); cudaFree(d_w); cudaFree(d_b);
    return -1;
}

int cuda_conv_transpose2d(
    const double* x, double* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const double* weight, const double* bias)
{
    int xn  = N * InC * H * W;
    int dn  = N * OutC * Hout * Wout;
    int wn  = InC * (OutC / groups) * KH * KW;
    int bn  = OutC;
    int inp = N * InC * H * W;

    double *d_x = NULL, *d_dst = NULL, *d_w = NULL, *d_b = NULL;
    if (cudaMalloc(&d_x,   xn * sizeof(double)) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, dn * sizeof(double)) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_w,   wn * sizeof(double)) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_b,   bn * sizeof(double)) != cudaSuccess) goto fail;

    if (cudaMemcpy(d_x, x,      xn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_w, weight, wn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    // Initialize dst with bias values on device.
    {
        // Fill with zeros first, then add bias per output channel.
        if (cudaMemset(d_dst, 0, dn * sizeof(double)) != cudaSuccess) goto fail;
        // Copy bias to host-local tmp and do a simple host-side expansion is not ideal;
        // use a small initialization kernel instead.
        // For correctness we just copy bias-initialized data from host.
        double* h_init = (double*)malloc(dn * sizeof(double));
        if (!h_init) goto fail;
        for (int ni = 0; ni < N; ni++) {
            for (int oc = 0; oc < OutC; oc++) {
                double b = bias[oc];
                int base = ni * OutC * Hout * Wout + oc * Hout * Wout;
                for (int i = 0; i < Hout * Wout; i++) h_init[base + i] = b;
            }
        }
        cudaMemcpy(d_dst, h_init, dn * sizeof(double), cudaMemcpyHostToDevice);
        free(h_init);
    }

    conv_transpose2d_kernel<<<blocks(inp), BLOCK>>>(
        d_x, d_dst, N, InC, H, W, OutC, KH, KW,
        sH, sW, pH, pW, dH, dW, groups, Hout, Wout, d_w);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, dn * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_x); cudaFree(d_dst); cudaFree(d_w); cudaFree(d_b);
    return 0;
fail:
    cudaFree(d_x); cudaFree(d_dst); cudaFree(d_w); cudaFree(d_b);
    return -1;
}

} // extern "C"
