#ifndef XLA_CONVOLUTION_H
#define XLA_CONVOLUTION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the PJRT client for the given platform ("cpu" or "gpu").
// Returns 0 on success, -1 on error.
// Note: shares the same PJRT client as other XLA backends if called from the
// same process — caller is responsible for serializing xla_init calls.
int xla_conv_init(const char* platform);

// Free all convolution XLA resources.
void xla_conv_shutdown(void);

// Conv1d forward.
// x[N*InC*L], weight[OutC*(InC/groups)*K], bias[OutC] -> dst[N*OutC*L_out]
int xla_conv1d(
    const double* x, double* dst,
    int N, int InC, int L,
    int OutC, int K,
    int stride, int pad, int dilation, int groups, int L_out,
    const double* weight, const double* bias);

// Conv2d forward.
int xla_conv2d(
    const double* x, double* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const double* weight, const double* bias);

// Conv3d forward.
int xla_conv3d(
    const double* x, double* dst,
    int N, int InC, int D, int H, int W,
    int OutC, int KD, int KH, int KW,
    int sD, int sH, int sW, int pD, int pH, int pW,
    int dD, int dH, int dW, int groups,
    int Dout, int Hout, int Wout,
    const double* weight, const double* bias);

// ConvTranspose2d forward.
int xla_conv_transpose2d(
    const double* x, double* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const double* weight, const double* bias);

#ifdef __cplusplus
}
#endif

#endif /* XLA_CONVOLUTION_H */
