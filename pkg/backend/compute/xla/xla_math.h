#ifndef PKG_BACKEND_COMPUTE_XLA_XLA_MATH_H
#define PKG_BACKEND_COMPUTE_XLA_XLA_MATH_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * xla_math_init(const char* platform), xla_math_shutdown(void):
 * - platform: NUL-terminated PJRT plugin selector (e.g. "cpu"); must not be NULL; behaviour if NULL is undefined.
 * - xla_math_init: returns 0 on success, -1 on failure (plugin load / client create). Not idempotent in the sense
 *   that repeated successful inits without shutdown may leak resources; callers should pair init/shutdown.
 * - xla_math_shutdown: destroys compiled executables and client; void return. Using math ops after shutdown without
 *   re-init is undefined (typically crashes or -1 from ops). Concurrent init/shutdown vs ops is undefined unless
 *   callers serialize externally.
 */

/** Host PJRT init for math kernels; see block comment above. */
int xla_math_init(const char* platform);

/** Releases PJRT client and cached executables; see block comment above. */
void xla_math_shutdown(void);

/**
 * xla_matmul: Row-major double GEMM. A is M×K, B is K×N, C is M×N (caller pre-allocates C).
 * Inputs must not overlap C (no in-place). Aliasing between A and B is undefined.
 * Returns 0 on success; -1 if M, K, or N are non-positive (implementation validates).
 */
int xla_matmul(const double* A, const double* B, double* C, int M, int K, int N);

/** Elementwise a+b -> out; length n; dst may alias inputs only if documented otherwise — treat as no-alias. */
int xla_add(const double* a, const double* b, double* out, int n);
/** Elementwise a*b -> out; length n. */
int xla_mul(const double* a, const double* b, double* out, int n);

/**
 * Elementwise dst[i] = src[i] / sqrt(dim). Arrays have length n.
 * dim must be positive; otherwise returns -1 and leaves dst untouched.
 */
int xla_inv_sqrt_dim_scale(const double* src, double* dst, int n, int dim);

/** Elementwise exp/log; IEEE-754 semantics; negative xla_log inputs yield NaN; log(0) yields -Inf; no validation. */
int xla_exp(const double* src, double* dst, int n);
/** See xla_exp/xla_log note above for domain. */
int xla_log(const double* src, double* dst, int n);

/**
 * xla_softmax: Row-major matrix src[num_rows][dim_size]. For each row i, reads dim_size doubles at src+i*dim_size,
 * stable softmax, writes to dst+i*dim_size. src/dst length must be num_rows*dim_size. src and dst must not alias
 * (in-place is unsupported). NaN/Inf follow typical exponential-sum softmax behaviour.
 */
int xla_softmax(const double* src, double* dst, int num_rows, int dim_size);

/**
 * xla_layernorm: Row-major src/dst length num_rows*d_model. Per row, normalize across d_model with eps (try 1e-5..1e-6).
 * weight/bias must be non-NULL in current implementation (length d_model). Returns 0 or -1 from inner failures.
 */
int xla_layernorm(
	const double* src, double* dst,
	const double* weight, const double* bias,
	int num_rows, int d_model, double eps
);

/** RMS norm per row; weight length d_model; non-NULL weight required by implementation. */
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

/**
 * xla_div_vec: Elementwise a[i]/b[i]. No zero check; IEEE-754 divide-by-zero yields ±Inf; returns 0 on completion,
 * -1 if StableHLO path fails before fallback.
 */
int xla_div_vec(const double* a, const double* b, double* dst, int n);

/**
 * xla_clamp_vec: Clamps dst[i] to [lo, hi]. Requires lo <= hi; if lo > hi returns -1 and leaves dst unchanged.
 */
int xla_clamp_vec(double* dst, double lo, double hi, int n);

#ifdef __cplusplus
}
#endif

#endif /* PKG_BACKEND_COMPUTE_XLA_XLA_MATH_H */
