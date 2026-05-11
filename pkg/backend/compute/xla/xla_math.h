#ifndef CARAMBA_XLA_MATH_H
#define CARAMBA_XLA_MATH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int xla_math_init(const char* platform);
void xla_math_shutdown(void);

int xla_matmul(const double* A, const double* B, double* C, int M, int K, int N);
int xla_add(const double* a, const double* b, double* out, int n);
int xla_mul(const double* a, const double* b, double* out, int n);
int xla_inv_sqrt_dim_scale(const double* src, double* dst, int n, int dim);
int xla_exp(const double* src, double* dst, int n);
int xla_log(const double* src, double* dst, int n);
int xla_softmax(const double* src, double* dst, int num_rows, int dim_size);
int xla_layernorm(
	const double* src, double* dst,
	const double* weight, const double* bias,
	int num_rows, int d_model, double eps
);
int xla_rmsnorm(
	const double* src, double* dst,
	const double* weight,
	int num_rows, int d_model, double eps
);
int xla_sign(const double* src, double* dst, int n);
int xla_outer(const double* a, const double* b, double* dst, int M, int N);

int xla_axpy(double* dst, const double* src, double scale, int n);
int xla_scale(double* dst, double s, int n);
int xla_sqrt_vec(const double* src, double* dst, int n);
int xla_add_scalar(double* dst, double scalar, int n);
int xla_div_vec(const double* a, const double* b, double* dst, int n);
int xla_clamp_vec(double* dst, double lo, double hi, int n);

#ifdef __cplusplus
}
#endif

#endif /* CARAMBA_XLA_MATH_H */
