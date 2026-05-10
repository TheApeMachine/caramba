#ifndef METAL_POOLING_H
#define METAL_POOLING_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal device, command queue, and pooling compute pipelines.
// metallib_path: path to the compiled pooling.metallib file.
// Returns 0 on success, -1 on failure.
int metal_pooling_init(const char* metallib_path);

// max_pool2d: 2-D max pooling over float32 input [N,C,H,W].
//   src      — host float* of length N*C*H*W
//   dst      — host float* of length N*C*Hout*Wout (caller-allocated)
//   N,C,H,W  — input shape
//   kH,kW    — kernel size
//   sH,sW    — stride
//   pH,pW    — padding
//   dH,dW    — dilation
//   Hout,Wout— output spatial dims
// Returns 0 on success, -1 on error.
int metal_max_pool2d(
    const float* src, float* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout);

// avg_pool2d: 2-D average pooling over float32 input [N,C,H,W].
//   divisor_override — if 0 use valid-element count; if >0 use this value
//   count_include_pad — 1: count padded zeros in divisor
int metal_avg_pool2d(
    const float* src, float* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout,
    int count_include_pad, int divisor_override);

// adaptive_avg_pool2d: output shape [N,C,OutH,OutW].
int metal_adaptive_avg_pool2d(
    const float* src, float* dst,
    int N, int C, int H, int W,
    int OutH, int OutW);

// adaptive_max_pool2d: output shape [N,C,OutH,OutW].
int metal_adaptive_max_pool2d(
    const float* src, float* dst,
    int N, int C, int H, int W,
    int OutH, int OutW);

#ifdef __cplusplus
}
#endif

#endif /* METAL_POOLING_H */
