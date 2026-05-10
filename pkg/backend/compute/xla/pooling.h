#ifndef XLA_POOLING_H
#define XLA_POOLING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the PJRT client (reuses xla_init from activation.h if already called).
// Returns 0 on success, -1 on error.
int xla_pooling_init(const char* platform);

// Compile pooling executables for the given spatial dimensions.
// Must be called before dispatch functions when dimensions change.
int xla_compile_pooling(
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout);

// max_pool2d: src has N*C*H*W doubles; dst has N*C*Hout*Wout doubles.
int xla_max_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout);

// avg_pool2d
int xla_avg_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout,
    int count_include_pad, int divisor_override);

// adaptive_avg_pool2d
int xla_adaptive_avg_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int OutH, int OutW);

// adaptive_max_pool2d
int xla_adaptive_max_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int OutH, int OutW);

// Free pooling PJRT resources.
void xla_pooling_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* XLA_POOLING_H */
