#ifndef PKG_BACKEND_COMPUTE_XLA_XLA_ACTIVE_INFERENCE_H
#define PKG_BACKEND_COMPUTE_XLA_XLA_ACTIVE_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * xla_ai_init / xla_ai_shutdown:
 * Initialise PJRT client for the given platform ("cpu"/"gpu").
 * Returns 0 on success, -1 on failure.
 */
int xla_ai_init(const char* platform);
void xla_ai_shutdown(void);

/**
 * xla_ai_free_energy:
 * F = 0.5 * sum(mu[i]^2 + exp(log_sigma[i]) - log_sigma[i] - 1).
 * Writes scalar to *out. Returns 0 on success.
 */
int xla_ai_free_energy(const double* mu, const double* log_sigma, double* out, int n);

/**
 * xla_ai_belief_update:
 *   out[0..n-1]  = mu - lr*(mu + pred_err)
 *   out[n..2n-1] = log_sigma - lr*(exp(log_sigma) - 1)
 * out must be pre-allocated to 2*n doubles.
 */
int xla_ai_belief_update(
    const double* mu, const double* log_sigma,
    const double* pred_err, double lr,
    double* out, int n);

/**
 * xla_ai_precision_weight:
 * out[i] = err[i] * exp(log_prec[i]). Length n.
 */
int xla_ai_precision_weight(
    const double* err, const double* log_prec, double* out, int n);

/**
 * xla_ai_expected_free_energy:
 * G[k] = -sum_i q[i,k]*ln(q[i,k]+eps)  for k in [0,K).
 * q_outcomes is [n*K] row-major (i stride K); out is [K].
 * eps must be finite and > 0 (e.g. 1e-12).
 */
int xla_ai_expected_free_energy(
    const double* q_outcomes, double* out, int n, int K, double eps);

#ifdef __cplusplus
}
#endif

#endif /* PKG_BACKEND_COMPUTE_XLA_XLA_ACTIVE_INFERENCE_H */
