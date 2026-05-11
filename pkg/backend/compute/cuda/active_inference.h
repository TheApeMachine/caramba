#ifndef CUDA_ACTIVE_INFERENCE_H
#define CUDA_ACTIVE_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

// Variational free energy F = 0.5*sum(mu^2 + exp(log_sigma) - log_sigma - 1).
// Returns scalar via *out. n = length of mu/log_sigma.
int cuda_ai_free_energy(const double* mu, const double* log_sigma, double* out, int n);

// Gradient descent step on F:
//   mu_new[i]        = mu[i]        - lr*(mu[i]+pred_err[i])
//   log_sigma_new[i] = log_sigma[i] - lr*(exp(log_sigma[i])-1)
// out must be pre-allocated to 2*n. First n = mu_new, next n = log_sigma_new.
int cuda_ai_belief_update(
    const double* mu, const double* log_sigma,
    const double* pred_err, double lr,
    double* out, int n);

// Precision-weighted prediction error: out[i] = err[i] * exp(log_prec[i]).
int cuda_ai_precision_weight(
    const double* err, const double* log_prec, double* out, int n);

// Expected free energy G[k] = -sum_i q[i,k]*ln(q[i,k]+eps) for k in [0,K).
// q_outcomes is [n*K] row-major (q[i,k] = q_outcomes[i*K+k]).
int cuda_ai_expected_free_energy(
    const double* q_outcomes, double* out, int n, int K);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_ACTIVE_INFERENCE_H */
