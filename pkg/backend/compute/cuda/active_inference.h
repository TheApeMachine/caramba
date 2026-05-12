#ifndef CUDA_ACTIVE_INFERENCE_H
#define CUDA_ACTIVE_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

// Variational free energy F = 0.5*sum(mu^2 + exp(log_sigma) - log_sigma - 1).
// Returns scalar via *out. n = length of mu/log_sigma.
// For n <= 0 returns -1; for n == 0 sets *out = 0 and returns 0.
int cuda_ai_free_energy(const double* mu, const double* log_sigma, double* out, int n);

// Gradient descent step on F:
//   mu_new[i]        = mu[i]        - lr*(mu[i]+pred_err[i])
//   log_sigma_new[i] = log_sigma[i] - lr*(exp(log_sigma[i])-1)
// out must be pre-allocated to 2*n. First n = mu_new, next n = log_sigma_new.
// Stability: keep log_sigma in a moderate range (e.g. roughly [-20, 20]) so exp(log_sigma)
// stays finite; very large positive values can overflow exp() and produce inf/NaN in out.
// lr should be positive; typical magnitudes are small (e.g. 1e-4 .. 1e-1).
// n <= 0 returns -1; n == 0 is a no-op (return 0).
int cuda_ai_belief_update(
    const double* mu, const double* log_sigma,
    const double* pred_err, double lr,
    double* out, int n);

// Precision-weighted prediction error: out[i] = err[i] * exp(log_prec[i]).
// Caller must pre-allocate out with at least n elements (out[i] for i in [0,n)).
// Large positive log_prec can make exp(log_prec) overflow to inf; clamp log_prec or
// compute in log-domain in a future revision if needed.
// n <= 0 returns -1; n == 0 is a no-op.
int cuda_ai_precision_weight(
    const double* err, const double* log_prec, double* out, int n);

// Expected free energy G[k] = -sum_i q[i,k]*ln(q[i,k]+eps) for k in [0,K).
// eps is fixed at 1e-12 inside the implementation (numerical stability in log).
// q_outcomes is row-major with n rows and K columns: q[i,k] = q_outcomes[i*K+k].
// out must be pre-allocated with K elements. Rows of q should be non-negative and
// approximately sum to 1 (within epsilon) for a well-defined entropy term.
// Returns non-zero if dimensions are invalid or CUDA fails.
int cuda_ai_expected_free_energy(
    const double* q_outcomes, double* out, int n, int K);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_ACTIVE_INFERENCE_H */
