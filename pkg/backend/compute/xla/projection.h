#ifndef XLA_PROJECTION_H
#define XLA_PROJECTION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types (re-declared here to avoid pulling in full XLA headers).
typedef struct PJRT_Client           PJRT_Client;
typedef struct PJRT_Buffer           PJRT_Buffer;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Error            PJRT_Error;

// Initialize the PJRT projection backend on the given platform ("cpu"/"gpu").
// Reuses the global g_api/g_client from activation_xla.cc if already initialised.
// Returns 0 on success, -1 on error.
int xla_projection_init(const char* platform);

// Compile StableHLO executables for the given matrix dimensions.
int xla_compile_projections(int M, int K, int N);

// Linear: dst[M*N] = src[M*K] @ weight[N*K]^T + bias[N]  (bias may be NULL)
int xla_linear(const double* src, const double* weight, const double* bias,
               double* dst, int M, int K, int N, int has_bias);

// FusedQKV: identical to xla_linear (split is done in Go).
int xla_fused_qkv(const double* src, const double* weight, const double* bias,
                  double* dst, int M, int K, int N, int has_bias);

// TiedEmbedding: dst[M*V] = src[M*D] @ weight[V*D]^T
int xla_tied_embedding(const double* src, const double* weight,
                       double* dst, int M, int D, int V);

// SPD: inv ≈ (L L^T + ridge I)^{-1} via Cholesky, triangular_solve, ZᵀZ on device.
// inv_out is n×n row-major; a is not modified.
int xla_spd_inverse(const double* a, int n, double ridge, double* inv_out);

// log|A + ridge I| for SPD (2 Σ log diag L).
int xla_spd_log_det(const double* a, int n, double ridge, double* log_det_out);

void xla_projection_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* XLA_PROJECTION_H */
