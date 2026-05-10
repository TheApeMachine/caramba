// Compile with:
// xcrun -sdk macosx metal -c pooling.metal -o pooling.air && xcrun -sdk macosx metallib pooling.air -o pooling.metallib

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Pooling parameter structs passed as constant buffers
// ---------------------------------------------------------------------------

struct MaxPoolParams {
    int N, C, H, W;
    int kH, kW, sH, sW;
    int pH, pW, dH, dW;
    int Hout, Wout;
};

struct AvgPoolParams {
    int N, C, H, W;
    int kH, kW, sH, sW;
    int pH, pW, dH, dW;
    int Hout, Wout;
    int count_include_pad;
    int divisor_override;
};

struct AdaptivePoolParams {
    int N, C, H, W;
    int OutH, OutW;
};

// ---------------------------------------------------------------------------
// max_pool2d_kernel
// One thread per output element. gid encodes linear index in [N*C*Hout*Wout].
// ---------------------------------------------------------------------------
kernel void max_pool2d_kernel(
    device const float*          src    [[buffer(0)]],
    device float*                dst    [[buffer(1)]],
    constant MaxPoolParams&      p      [[buffer(2)]],
    uint                         gid    [[thread_position_in_grid]])
{
    int total = p.N * p.C * p.Hout * p.Wout;
    if ((int)gid >= total) return;

    int tmp = (int)gid;
    int ow  = tmp % p.Wout; tmp /= p.Wout;
    int oh  = tmp % p.Hout; tmp /= p.Hout;
    int c   = tmp % p.C;    tmp /= p.C;
    int n   = tmp;

    int hStart = oh * p.sH - p.pH;
    int wStart = ow * p.sW - p.pW;
    int baseIn = (n * p.C + c) * p.H * p.W;

    float maxVal = -INFINITY;
    for (int kh = 0; kh < p.kH; kh++) {
        int ih = hStart + kh * p.dH;
        if (ih < 0 || ih >= p.H) continue;
        for (int kw = 0; kw < p.kW; kw++) {
            int iw = wStart + kw * p.dW;
            if (iw < 0 || iw >= p.W) continue;
            float v = src[baseIn + ih * p.W + iw];
            if (v > maxVal) maxVal = v;
        }
    }
    dst[(int)gid] = maxVal;
}

// ---------------------------------------------------------------------------
// avg_pool2d_kernel
// ---------------------------------------------------------------------------
kernel void avg_pool2d_kernel(
    device const float*          src    [[buffer(0)]],
    device float*                dst    [[buffer(1)]],
    constant AvgPoolParams&      p      [[buffer(2)]],
    uint                         gid    [[thread_position_in_grid]])
{
    int total = p.N * p.C * p.Hout * p.Wout;
    if ((int)gid >= total) return;

    int tmp = (int)gid;
    int ow  = tmp % p.Wout; tmp /= p.Wout;
    int oh  = tmp % p.Hout; tmp /= p.Hout;
    int c   = tmp % p.C;    tmp /= p.C;
    int n   = tmp;

    int hStart = oh * p.sH - p.pH;
    int wStart = ow * p.sW - p.pW;
    int baseIn = (n * p.C + c) * p.H * p.W;

    float sum = 0.0f;
    int validCount = 0;
    int kernelCount = 0;
    for (int kh = 0; kh < p.kH; kh++) {
        int ih = hStart + kh * p.dH;
        bool validH = (ih >= 0 && ih < p.H);
        for (int kw = 0; kw < p.kW; kw++) {
            kernelCount++;
            int iw = wStart + kw * p.dW;
            if (validH && iw >= 0 && iw < p.W) {
                sum += src[baseIn + ih * p.W + iw];
                validCount++;
            }
        }
    }

    int divisor;
    if (p.divisor_override != 0) {
        divisor = p.divisor_override;
    } else if (p.count_include_pad) {
        divisor = kernelCount;
    } else {
        divisor = validCount;
    }

    dst[(int)gid] = (divisor > 0) ? sum / (float)divisor : 0.0f;
}

// ---------------------------------------------------------------------------
// adaptive_avg_pool2d_kernel
// ---------------------------------------------------------------------------
kernel void adaptive_avg_pool2d_kernel(
    device const float*            src    [[buffer(0)]],
    device float*                  dst    [[buffer(1)]],
    constant AdaptivePoolParams&   p      [[buffer(2)]],
    uint                           gid    [[thread_position_in_grid]])
{
    int total = p.N * p.C * p.OutH * p.OutW;
    if ((int)gid >= total) return;

    int tmp = (int)gid;
    int ow  = tmp % p.OutW; tmp /= p.OutW;
    int oh  = tmp % p.OutH; tmp /= p.OutH;
    int c   = tmp % p.C;    tmp /= p.C;
    int n   = tmp;

    int hStart = oh * p.H / p.OutH;
    int hEnd   = ((oh + 1) * p.H + p.OutH - 1) / p.OutH;
    int wStart = ow * p.W / p.OutW;
    int wEnd   = ((ow + 1) * p.W + p.OutW - 1) / p.OutW;

    int baseIn = (n * p.C + c) * p.H * p.W;
    float sum = 0.0f;
    int cnt = (hEnd - hStart) * (wEnd - wStart);
    for (int ih = hStart; ih < hEnd; ih++) {
        for (int iw = wStart; iw < wEnd; iw++) {
            sum += src[baseIn + ih * p.W + iw];
        }
    }
    dst[(int)gid] = (cnt > 0) ? sum / (float)cnt : 0.0f;
}

// ---------------------------------------------------------------------------
// adaptive_max_pool2d_kernel
// ---------------------------------------------------------------------------
kernel void adaptive_max_pool2d_kernel(
    device const float*            src    [[buffer(0)]],
    device float*                  dst    [[buffer(1)]],
    constant AdaptivePoolParams&   p      [[buffer(2)]],
    uint                           gid    [[thread_position_in_grid]])
{
    int total = p.N * p.C * p.OutH * p.OutW;
    if ((int)gid >= total) return;

    int tmp = (int)gid;
    int ow  = tmp % p.OutW; tmp /= p.OutW;
    int oh  = tmp % p.OutH; tmp /= p.OutH;
    int c   = tmp % p.C;    tmp /= p.C;
    int n   = tmp;

    int hStart = oh * p.H / p.OutH;
    int hEnd   = ((oh + 1) * p.H + p.OutH - 1) / p.OutH;
    int wStart = ow * p.W / p.OutW;
    int wEnd   = ((ow + 1) * p.W + p.OutW - 1) / p.OutW;

    int baseIn = (n * p.C + c) * p.H * p.W;
    float maxVal = -INFINITY;
    for (int ih = hStart; ih < hEnd; ih++) {
        for (int iw = wStart; iw < wEnd; iw++) {
            float v = src[baseIn + ih * p.W + iw];
            if (v > maxVal) maxVal = v;
        }
    }
    dst[(int)gid] = maxVal;
}
