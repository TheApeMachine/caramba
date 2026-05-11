#ifndef PKG_BACKEND_COMPUTE_XLA_XLA_DYNAMIC_CODING_H
#define PKG_BACKEND_COMPUTE_XLA_XLA_DYNAMIC_CODING_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * xla_dc_init / xla_dc_shutdown:
 * Initialise PJRT client for the given platform ("cpu"/"gpu").
 * Returns 0 on success, -1 on failure.
 */
int xla_dc_init(const char* platform);
void xla_dc_shutdown(void);

/**
 * xla_dc_project: dst[dOut] = W[dOut*dIn] @ x[dIn]. Row-major double.
 * Returns 0 on success, -1 on invalid dimensions or runtime error.
 */
int xla_dc_project(const double* W, const double* x, double* dst, int dOut, int dIn);

/**
 * xla_dc_reconstruct: dst[dIn] = W^T[dIn*dOut] @ z[dOut]. Row-major double.
 */
int xla_dc_reconstruct(const double* W, const double* z, double* dst, int dIn, int dOut);

/**
 * xla_dc_dynamics: dst[d] = A[d*d]@z[d] + B[d*du]@u[du]. Row-major double.
 */
int xla_dc_dynamics(
    const double* z, const double* A,
    const double* u, const double* B,
    double* dst, int d, int du);

/**
 * xla_dc_manifold_distance: *dist = sqrt(sum((a[i]-b[i])^2)). Length d.
 */
int xla_dc_manifold_distance(const double* a, const double* b, double* dist, int d);

#ifdef __cplusplus
}
#endif

#endif /* PKG_BACKEND_COMPUTE_XLA_XLA_DYNAMIC_CODING_H */
