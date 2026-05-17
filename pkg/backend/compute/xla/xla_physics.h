#ifndef PKG_BACKEND_COMPUTE_XLA_XLA_PHYSICS_H
#define PKG_BACKEND_COMPUTE_XLA_XLA_PHYSICS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * XLA physics-stencil backend via the PJRT C API.
 *
 * Lifecycle mirrors xla_math: call xla_physics_init(platform) before any
 * kernel, xla_physics_shutdown() at process teardown. Repeated init without
 * shutdown leaks resources; concurrent init/shutdown vs ops is undefined.
 */

/** Host PJRT init for physics kernels. Returns 0 on success, -1 on failure. */
int xla_physics_init(const char* platform);

/** Releases PJRT client and cached executables. */
void xla_physics_shutdown(void);

/**
 * xla_laplacian_1d: 2nd-order central-difference Laplacian on a uniform
 * 1D periodic grid of length n.
 *   dst[i] = (src[(i-1+n)%n] + src[(i+1)%n] - 2*src[i]) * inv_h2
 *
 * Requires n >= 2; the caller handles the degenerate n==1 case scalar
 * (output is identically zero because both neighbours wrap to the same cell).
 * src and dst length must be n. src and dst must not alias.
 * Returns 0 on success, -1 on validation or PJRT failure.
 */
int xla_laplacian_1d(const double* src, double* dst, int n, double inv_h2);

/**
 * xla_laplacian_2d: periodic 5-point Laplacian on row-major [H, W] grid.
 * Requires H >= 2 and W >= 2.
 */
int xla_laplacian_2d(const double* src, double* dst, int H, int W, double inv_h2);

/**
 * xla_laplacian_3d: periodic 7-point Laplacian on row-major [D, H, W] grid.
 * Requires D >= 2, H >= 2, W >= 2.
 */
int xla_laplacian_3d(const double* src, double* dst, int D, int H, int W, double inv_h2);

#ifdef __cplusplus
}
#endif

#endif /* PKG_BACKEND_COMPUTE_XLA_XLA_PHYSICS_H */
