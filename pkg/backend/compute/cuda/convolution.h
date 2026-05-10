#ifndef CUDA_CONVOLUTION_H
#define CUDA_CONVOLUTION_H

#ifdef __cplusplus
extern "C" {
#endif

// All functions take host double* pointers; device memory is managed internally.
// Returns 0 on success, -1 on CUDA error.

// Conv1d: x[N,InC,L], weight[OutC, InC/groups, K], bias[OutC] -> dst[N,OutC,L_out]
int cuda_conv1d(
    const double* x, double* dst,
    int N, int InC, int L,
    int OutC, int K,
    int stride, int pad, int dilation, int groups,
    int L_out,
    const double* weight, const double* bias);

// Conv2d: x[N,InC,H,W], weight[OutC,InC/groups,KH,KW], bias[OutC] -> dst[N,OutC,Hout,Wout]
int cuda_conv2d(
    const double* x, double* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const double* weight, const double* bias);

// Conv3d: x[N,InC,D,H,W], weight[OutC,InC/groups,KD,KH,KW], bias[OutC]
int cuda_conv3d(
    const double* x, double* dst,
    int N, int InC, int D, int H, int W,
    int OutC, int KD, int KH, int KW,
    int sD, int sH, int sW, int pD, int pH, int pW,
    int dD, int dH, int dW, int groups,
    int Dout, int Hout, int Wout,
    const double* weight, const double* bias);

// ConvTranspose2d: x[N,InC,H,W], weight[InC,OutC/groups,KH,KW], bias[OutC]
// Uses atomicAdd; requires sm_60+.
int cuda_conv_transpose2d(
    const double* x, double* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const double* weight, const double* bias);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_CONVOLUTION_H */
