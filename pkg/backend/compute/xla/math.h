#ifndef XLA_MATH_H
#define XLA_MATH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the PJRT client for math ops (same platform as activation).
// platform: "cpu" or "gpu". Returns 0 on success, -1 on error.
int xla_math_init(const char* platform);

// Shutdown math resources.
void xla_math_shutdown(void);

// Matrix multiply: C[M*N] = A[M*K] x B[K*N]
int xla_matmul(const double* A, const double* B, double* C, int M, int K, int N);

// Elementwise add
int xla_add(const double* a, const double* b, double* out, int n);

// Elementwise multiply
int xla_mul(const double* a, const double* b, double* out, int n);

// Scale: dst = src * (1/sqrt(dim))
int xla_inv_sqrt_dim_scale(const double* src, double* dst, int n, int dim);

// Elementwise exp
int xla_exp(const double* src, double* dst, int n);

// Elementwise log
int xla_log(const double* src, double* dst, int n);

// Softmax over last dim
int xla_softmax(const double* src, double* dst, int num_rows, int dim_size);

// Layer norm
int xla_layernorm(const double* src, double* dst,
                  const double* weight, const double* bias,
                  int num_rows, int d_model, double eps);

// RMS norm
int xla_rmsnorm(const double* src, double* dst,
                const double* weight,
                int num_rows, int d_model, double eps);

// Elementwise sign: dst[i] = +1 if src[i] > 0, -1 if < 0, 0 if == 0.
int xla_sign(const double* src, double* dst, int n);

// Outer product: dst[i*N+j] = a[i] * b[j].
int xla_outer(const double* a, const double* b, double* dst, int M, int N);

// Optimizer primitives
int xla_axpy(double* dst, const double* src, double scale, int n);
int xla_scale(double* dst, double s, int n);
int xla_sqrt_vec(const double* src, double* dst, int n);
int xla_add_scalar(double* dst, double scalar, int n);
int xla_div_vec(const double* a, const double* b, double* dst, int n);
int xla_clamp_vec(double* dst, double lo, double hi, int n);

#ifdef __cplusplus
}
#endif

#endif /* XLA_MATH_H */
