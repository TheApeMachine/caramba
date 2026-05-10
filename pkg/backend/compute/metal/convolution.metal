// Compile with:
// xcrun -sdk macosx metal -c convolution.metal -o convolution.air && xcrun -sdk macosx metallib convolution.air -o convolution.metallib

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Conv1d: each thread computes one output element [n, oc, l_out]
// ---------------------------------------------------------------------------

struct Conv1dParams {
    int N, InC, L;
    int OutC, K;
    int stride, pad, dilation, groups;
    int L_out;
};

kernel void conv1d_kernel(
    device const float* x       [[buffer(0)]],
    device float*       dst     [[buffer(1)]],
    constant Conv1dParams& p    [[buffer(2)]],
    device const float* weight  [[buffer(3)]],
    device const float* bias    [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]])
{
    int total = p.N * p.OutC * p.L_out;
    if ((int)gid >= total) return;

    int wo     = (int)gid % p.L_out;
    int tmp    = (int)gid / p.L_out;
    int oc     = tmp % p.OutC;
    int ni     = tmp / p.OutC;

    int g           = oc / (p.OutC / p.groups);
    int ocPerGroup  = p.OutC / p.groups;
    int icPerGroup  = p.InC  / p.groups;
    int ocLocal     = oc % ocPerGroup;
    int icStart     = g * icPerGroup;

    int kernElems   = icPerGroup * p.K;
    device const float* wRow = weight + (g * ocPerGroup + ocLocal) * kernElems;

    float sum = bias[oc];
    for (int ic = 0; ic < icPerGroup; ic++) {
        int absIC = icStart + ic;
        for (int k = 0; k < p.K; k++) {
            int li = wo * p.stride + k * p.dilation - p.pad;
            if (li >= 0 && li < p.L) {
                sum += x[ni * p.InC * p.L + absIC * p.L + li] * wRow[ic * p.K + k];
            }
        }
    }
    dst[ni * p.OutC * p.L_out + oc * p.L_out + wo] = sum;
}

// ---------------------------------------------------------------------------
// Conv2d: each thread computes one output element [n, oc, h_out, w_out]
// ---------------------------------------------------------------------------

struct Conv2dParams {
    int N, InC, H, W;
    int OutC, KH, KW;
    int sH, sW, pH, pW, dH, dW, groups;
    int Hout, Wout;
};

kernel void conv2d_kernel(
    device const float* x       [[buffer(0)]],
    device float*       dst     [[buffer(1)]],
    constant Conv2dParams& p    [[buffer(2)]],
    device const float* weight  [[buffer(3)]],
    device const float* bias    [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]])
{
    int total = p.N * p.OutC * p.Hout * p.Wout;
    if ((int)gid >= total) return;

    int wo     = (int)gid % p.Wout;
    int tmp    = (int)gid / p.Wout;
    int ho     = tmp % p.Hout;
    tmp        = tmp / p.Hout;
    int oc     = tmp % p.OutC;
    int ni     = tmp / p.OutC;

    int ocPerGroup  = p.OutC / p.groups;
    int icPerGroup  = p.InC  / p.groups;
    int g           = oc / ocPerGroup;
    int ocLocal     = oc % ocPerGroup;
    int icStart     = g * icPerGroup;

    int kernElems   = icPerGroup * p.KH * p.KW;
    device const float* wRow = weight + (g * ocPerGroup + ocLocal) * kernElems;

    float sum = bias[oc];
    for (int ic = 0; ic < icPerGroup; ic++) {
        int absIC = icStart + ic;
        for (int kh = 0; kh < p.KH; kh++) {
            int hi = ho * p.sH + kh * p.dH - p.pH;
            if (hi < 0 || hi >= p.H) continue;
            for (int kw = 0; kw < p.KW; kw++) {
                int wi = wo * p.sW + kw * p.dW - p.pW;
                if (wi >= 0 && wi < p.W) {
                    sum += x[ni*p.InC*p.H*p.W + absIC*p.H*p.W + hi*p.W + wi]
                         * wRow[ic*p.KH*p.KW + kh*p.KW + kw];
                }
            }
        }
    }
    dst[ni*p.OutC*p.Hout*p.Wout + oc*p.Hout*p.Wout + ho*p.Wout + wo] = sum;
}

// ---------------------------------------------------------------------------
// Conv3d: each thread computes one output element [n, oc, d_out, h_out, w_out]
// ---------------------------------------------------------------------------

struct Conv3dParams {
    int N, InC, D, H, W;
    int OutC, KD, KH, KW;
    int sD, sH, sW, pD, pH, pW, dD, dH, dW, groups;
    int Dout, Hout, Wout;
};

kernel void conv3d_kernel(
    device const float* x       [[buffer(0)]],
    device float*       dst     [[buffer(1)]],
    constant Conv3dParams& p    [[buffer(2)]],
    device const float* weight  [[buffer(3)]],
    device const float* bias    [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]])
{
    int total = p.N * p.OutC * p.Dout * p.Hout * p.Wout;
    if ((int)gid >= total) return;

    int wo  = (int)gid % p.Wout;
    int tmp = (int)gid / p.Wout;
    int ho  = tmp % p.Hout;
    tmp     = tmp / p.Hout;
    int doo = tmp % p.Dout;
    tmp     = tmp / p.Dout;
    int oc  = tmp % p.OutC;
    int ni  = tmp / p.OutC;

    int ocPerGroup  = p.OutC / p.groups;
    int icPerGroup  = p.InC  / p.groups;
    int g           = oc / ocPerGroup;
    int ocLocal     = oc % ocPerGroup;
    int icStart     = g * icPerGroup;

    int kernElems   = icPerGroup * p.KD * p.KH * p.KW;
    device const float* wRow = weight + (g * ocPerGroup + ocLocal) * kernElems;

    float sum = bias[oc];
    for (int ic = 0; ic < icPerGroup; ic++) {
        int absIC = icStart + ic;
        for (int kd = 0; kd < p.KD; kd++) {
            int di = doo * p.sD + kd * p.dD - p.pD;
            if (di < 0 || di >= p.D) continue;
            for (int kh = 0; kh < p.KH; kh++) {
                int hi = ho * p.sH + kh * p.dH - p.pH;
                if (hi < 0 || hi >= p.H) continue;
                for (int kw = 0; kw < p.KW; kw++) {
                    int wi = wo * p.sW + kw * p.dW - p.pW;
                    if (wi >= 0 && wi < p.W) {
                        int xIdx = ni*p.InC*p.D*p.H*p.W + absIC*p.D*p.H*p.W
                                 + di*p.H*p.W + hi*p.W + wi;
                        int wIdx = ic*p.KD*p.KH*p.KW + kd*p.KH*p.KW + kh*p.KW + kw;
                        sum += x[xIdx] * wRow[wIdx];
                    }
                }
            }
        }
    }
    dst[ni*p.OutC*p.Dout*p.Hout*p.Wout + oc*p.Dout*p.Hout*p.Wout
       + doo*p.Hout*p.Wout + ho*p.Wout + wo] = sum;
}

// ---------------------------------------------------------------------------
// ConvTranspose2d: each thread processes one input pixel [n, ic, h, w]
// and scatter-adds into the output.  Uses atomic_fetch_add_explicit.
// ---------------------------------------------------------------------------

struct ConvTranspose2dParams {
    int N, InC, H, W;
    int OutC, KH, KW;
    int sH, sW, pH, pW, dH, dW, groups;
    int Hout, Wout;
};

kernel void conv_transpose2d_kernel(
    device const float* x       [[buffer(0)]],
    device atomic_float* dst    [[buffer(1)]],
    constant ConvTranspose2dParams& p [[buffer(2)]],
    device const float* weight  [[buffer(3)]],
    device const float* bias    [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]])
{
    int total = p.N * p.InC * p.H * p.W;
    if ((int)gid >= total) return;

    int wi    = (int)gid % p.W;
    int tmp   = (int)gid / p.W;
    int hi    = tmp % p.H;
    tmp       = tmp / p.H;
    int ic    = tmp % p.InC;
    int ni    = tmp / p.InC;

    int ocPerGroup  = p.OutC / p.groups;
    int icPerGroup  = p.InC  / p.groups;
    int g           = ic / icPerGroup;
    int icLocal     = ic % icPerGroup;
    int ocStart     = g * ocPerGroup;

    int kernElems   = ocPerGroup * p.KH * p.KW;
    device const float* wRow = weight + ic * kernElems;

    float xVal = x[ni*p.InC*p.H*p.W + ic*p.H*p.W + hi*p.W + wi];

    for (int oc = 0; oc < ocPerGroup; oc++) {
        int absOC = ocStart + oc;
        for (int kh = 0; kh < p.KH; kh++) {
            int ho = hi * p.sH + kh * p.dH - p.pH;
            if (ho < 0 || ho >= p.Hout) continue;
            for (int kw = 0; kw < p.KW; kw++) {
                int wo = wi * p.sW + kw * p.dW - p.pW;
                if (wo >= 0 && wo < p.Wout) {
                    int oIdx = ni*p.OutC*p.Hout*p.Wout + absOC*p.Hout*p.Wout + ho*p.Wout + wo;
                    int wIdx = oc*p.KH*p.KW + kh*p.KW + kw;
                    atomic_fetch_add_explicit(&dst[oIdx], xVal * wRow[wIdx],
                                             memory_order_relaxed);
                }
            }
        }
    }
    // Add bias contribution — only thread responsible for (ni, absOC, 0, 0) does it,
    // but for simplicity bias is pre-filled by the host before kernel launch.
    (void)icLocal;
}
