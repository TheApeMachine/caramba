#include <cuda_runtime.h>
#include <float.h>
#include "pooling.h"

// ---------------------------------------------------------------------------
// Device kernels — double precision, one thread per output element.
// ---------------------------------------------------------------------------

__global__ void max_pool2d_kernel(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Hout * Wout;
    if (idx >= total) return;

    int tmp = idx;
    int ow  = tmp % Wout; tmp /= Wout;
    int oh  = tmp % Hout; tmp /= Hout;
    int c   = tmp % C;    tmp /= C;
    int n   = tmp;

    int hStart = oh * sH - pH;
    int wStart = ow * sW - pW;
    int baseIn = (n * C + c) * H * W;

    double maxVal = -DBL_MAX;
    for (int kh = 0; kh < kH; kh++) {
        int ih = hStart + kh * dH;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kW; kw++) {
            int iw = wStart + kw * dW;
            if (iw < 0 || iw >= W) continue;
            double v = src[baseIn + ih * W + iw];
            if (v > maxVal) maxVal = v;
        }
    }
    dst[idx] = maxVal;
}

__global__ void avg_pool2d_kernel(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout,
    int count_include_pad, int divisor_override)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Hout * Wout;
    if (idx >= total) return;

    int tmp = idx;
    int ow  = tmp % Wout; tmp /= Wout;
    int oh  = tmp % Hout; tmp /= Hout;
    int c   = tmp % C;    tmp /= C;
    int n   = tmp;

    int hStart = oh * sH - pH;
    int wStart = ow * sW - pW;
    int baseIn = (n * C + c) * H * W;

    double sum = 0.0;
    int validCount = 0;
    int kernelCount = 0;
    for (int kh = 0; kh < kH; kh++) {
        int ih = hStart + kh * dH;
        int validH = (ih >= 0 && ih < H);
        for (int kw = 0; kw < kW; kw++) {
            kernelCount++;
            int iw = wStart + kw * dW;
            if (validH && iw >= 0 && iw < W) {
                sum += src[baseIn + ih * W + iw];
                validCount++;
            }
        }
    }

    int divisor;
    if (divisor_override != 0) {
        divisor = divisor_override;
    } else if (count_include_pad) {
        divisor = kernelCount;
    } else {
        divisor = validCount;
    }

    dst[idx] = (divisor > 0) ? sum / (double)divisor : 0.0;
}

__global__ void adaptive_avg_pool2d_kernel(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int OutH, int OutW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * OutH * OutW;
    if (idx >= total) return;

    int tmp = idx;
    int ow  = tmp % OutW; tmp /= OutW;
    int oh  = tmp % OutH; tmp /= OutH;
    int c   = tmp % C;    tmp /= C;
    int n   = tmp;

    int hStart = oh * H / OutH;
    int hEnd   = (oh + 1) * H / OutH + ((((oh+1)*H) % OutH) != 0 ? 1 : 0);
    int wStart = ow * W / OutW;
    int wEnd   = (ow + 1) * W / OutW + ((((ow+1)*W) % OutW) != 0 ? 1 : 0);

    int baseIn = (n * C + c) * H * W;
    double sum = 0.0;
    int cnt = (hEnd - hStart) * (wEnd - wStart);
    for (int ih = hStart; ih < hEnd; ih++) {
        for (int iw = wStart; iw < wEnd; iw++) {
            sum += src[baseIn + ih * W + iw];
        }
    }
    dst[idx] = (cnt > 0) ? sum / (double)cnt : 0.0;
}

__global__ void adaptive_max_pool2d_kernel(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int OutH, int OutW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * OutH * OutW;
    if (idx >= total) return;

    int tmp = idx;
    int ow  = tmp % OutW; tmp /= OutW;
    int oh  = tmp % OutH; tmp /= OutH;
    int c   = tmp % C;    tmp /= C;
    int n   = tmp;

    int hStart = oh * H / OutH;
    int hEnd   = (oh + 1) * H / OutH + ((((oh+1)*H) % OutH) != 0 ? 1 : 0);
    int wStart = ow * W / OutW;
    int wEnd   = (ow + 1) * W / OutW + ((((ow+1)*W) % OutW) != 0 ? 1 : 0);

    int baseIn = (n * C + c) * H * W;
    double maxVal = -DBL_MAX;
    for (int ih = hStart; ih < hEnd; ih++) {
        for (int iw = wStart; iw < wEnd; iw++) {
            double v = src[baseIn + ih * W + iw];
            if (v > maxVal) maxVal = v;
        }
    }
    dst[idx] = maxVal;
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

static const int POOL_BLOCK = 256;
static inline int pool_blocks(int n) { return (n + POOL_BLOCK - 1) / POOL_BLOCK; }

// ---------------------------------------------------------------------------
// C linkage wrappers
// ---------------------------------------------------------------------------

extern "C" {

int cuda_max_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout)
{
    int src_n = N * C * H * W;
    int dst_n = N * C * Hout * Wout;
    size_t src_bytes = (size_t)src_n * sizeof(double);
    size_t dst_bytes = (size_t)dst_n * sizeof(double);

    double *d_src = NULL, *d_dst = NULL;
    if (cudaMalloc(&d_src, src_bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, dst_bytes) != cudaSuccess) { cudaFree(d_src); return -1; }
    if (cudaMemcpy(d_src, src, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    max_pool2d_kernel<<<pool_blocks(dst_n), POOL_BLOCK>>>(
        d_src, d_dst, N, C, H, W, kH, kW, sH, sW, pH, pW, dH, dW, Hout, Wout);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src); cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src); cudaFree(d_dst);
    return -1;
}

int cuda_avg_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout,
    int count_include_pad, int divisor_override)
{
    int src_n = N * C * H * W;
    int dst_n = N * C * Hout * Wout;
    size_t src_bytes = (size_t)src_n * sizeof(double);
    size_t dst_bytes = (size_t)dst_n * sizeof(double);

    double *d_src = NULL, *d_dst = NULL;
    if (cudaMalloc(&d_src, src_bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, dst_bytes) != cudaSuccess) { cudaFree(d_src); return -1; }
    if (cudaMemcpy(d_src, src, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    avg_pool2d_kernel<<<pool_blocks(dst_n), POOL_BLOCK>>>(
        d_src, d_dst, N, C, H, W, kH, kW, sH, sW, pH, pW, dH, dW, Hout, Wout,
        count_include_pad, divisor_override);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src); cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src); cudaFree(d_dst);
    return -1;
}

int cuda_adaptive_avg_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int OutH, int OutW)
{
    int src_n = N * C * H * W;
    int dst_n = N * C * OutH * OutW;
    size_t src_bytes = (size_t)src_n * sizeof(double);
    size_t dst_bytes = (size_t)dst_n * sizeof(double);

    double *d_src = NULL, *d_dst = NULL;
    if (cudaMalloc(&d_src, src_bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, dst_bytes) != cudaSuccess) { cudaFree(d_src); return -1; }
    if (cudaMemcpy(d_src, src, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    adaptive_avg_pool2d_kernel<<<pool_blocks(dst_n), POOL_BLOCK>>>(
        d_src, d_dst, N, C, H, W, OutH, OutW);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src); cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src); cudaFree(d_dst);
    return -1;
}

int cuda_adaptive_max_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int OutH, int OutW)
{
    int src_n = N * C * H * W;
    int dst_n = N * C * OutH * OutW;
    size_t src_bytes = (size_t)src_n * sizeof(double);
    size_t dst_bytes = (size_t)dst_n * sizeof(double);

    double *d_src = NULL, *d_dst = NULL;
    if (cudaMalloc(&d_src, src_bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_dst, dst_bytes) != cudaSuccess) { cudaFree(d_src); return -1; }
    if (cudaMemcpy(d_src, src, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    adaptive_max_pool2d_kernel<<<pool_blocks(dst_n), POOL_BLOCK>>>(
        d_src, d_dst, N, C, H, W, OutH, OutW);
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_src); cudaFree(d_dst);
    return 0;
fail:
    cudaFree(d_src); cudaFree(d_dst);
    return -1;
}

} // extern "C"
