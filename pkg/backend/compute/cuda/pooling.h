#ifndef CUDA_POOLING_H
#define CUDA_POOLING_H

#ifdef __cplusplus
extern "C" {
#endif

// All functions use double (float64) for inputs and outputs.
// src/dst are host pointers; device memory is managed internally.
// Returns 0 on success, -1 on CUDA error.

// max_pool2d: shape [N,C,H,W] -> [N,C,Hout,Wout]
int cuda_max_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout);

// avg_pool2d: count_include_pad=1 means padded zeros count in denominator.
//             divisor_override=0 means natural count.
int cuda_avg_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout,
    int count_include_pad, int divisor_override);

// adaptive_avg_pool2d: variable-region average pooling.
int cuda_adaptive_avg_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int OutH, int OutW);

// adaptive_max_pool2d: variable-region max pooling.
int cuda_adaptive_max_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int OutH, int OutW);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_POOLING_H */
