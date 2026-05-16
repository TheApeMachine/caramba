#ifndef METAL_CONVOLUTION_H
#define METAL_CONVOLUTION_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal device, command queue, and all convolution compute pipelines.
// metallib_path: path to the compiled convolution.metallib file.
// Returns 0 on success, -1 on failure.
int metal_conv_init(const char* metallib_path);

// Conv1d forward pass.
// x: input [N, InC, L], weight: [OutC, InC/groups, K], bias: [OutC]
// Returns 0 on success, -1 on failure.
int metal_conv1d(
    const float* x, float* dst,
    int N, int InC, int L,
    int OutC, int K,
    int stride, int pad, int dilation, int groups,
    int L_out,
    const float* weight, const float* bias);

int metal_conv1d_tensor(
    const void* x, void* dst,
    int N, int InC, int L,
    int OutC, int K,
    int stride, int pad, int dilation, int groups,
    int L_out,
    const void* weight, const void* bias);

// Conv2d forward pass.
// x: input [N, InC, H, W], weight: [OutC, InC/groups, KH, KW], bias: [OutC]
int metal_conv2d(
    const float* x, float* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const float* weight, const float* bias);

int metal_conv2d_tensor(
    const void* x, void* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const void* weight, const void* bias);

// Conv3d forward pass.
// x: input [N, InC, D, H, W], weight: [OutC, InC/groups, KD, KH, KW], bias: [OutC]
int metal_conv3d(
    const float* x, float* dst,
    int N, int InC, int D, int H, int W,
    int OutC, int KD, int KH, int KW,
    int sD, int sH, int sW, int pD, int pH, int pW,
    int dD, int dH, int dW, int groups,
    int Dout, int Hout, int Wout,
    const float* weight, const float* bias);

int metal_conv3d_tensor(
    const void* x, void* dst,
    int N, int InC, int D, int H, int W,
    int OutC, int KD, int KH, int KW,
    int sD, int sH, int sW, int pD, int pH, int pW,
    int dD, int dH, int dW, int groups,
    int Dout, int Hout, int Wout,
    const void* weight, const void* bias);

// ConvTranspose2d forward pass (scatter-add via atomics).
// x: input [N, InC, H, W], weight: [InC, OutC/groups, KH, KW], bias: [OutC]
// dst must be pre-zeroed by caller.
int metal_conv_transpose2d(
    const float* x, float* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const float* weight, const float* bias);

int metal_conv_transpose2d_tensor(
    const void* x, void* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const void* weight, const void* bias);

#ifdef __cplusplus
}
#endif

#endif /* METAL_CONVOLUTION_H */
