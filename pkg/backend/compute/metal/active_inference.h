#ifndef METAL_ACTIVE_INFERENCE_H
#define METAL_ACTIVE_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal device/queue and compile active inference pipelines.
// metallib_path: absolute path to active_inference.metallib.
// Returns 0 on success, -1 on failure.
int metal_ai_init(const char* metallib_path);

// Variational free energy F = 0.5*sum(mu^2 + exp(ls) - ls - 1).
// Returns scalar via *out. Float32. n = number of elements.
int metal_ai_free_energy(const float* mu, const float* log_sigma, float* out, int n);

// Belief update:
//   out[0..n-1]   = mu - lr*(mu + pred_err)
//   out[n..2n-1]  = log_sigma - lr*(exp(log_sigma) - 1)
// out must be pre-allocated to 2*n floats.
int metal_ai_belief_update(
    const float* mu, const float* log_sigma,
    const float* pred_err, float lr,
    float* out, int n);

// Precision-weighted error: out[i] = err[i] * exp(log_prec[i]). Float32.
int metal_ai_precision_weight(
    const float* err, const float* log_prec, float* out, int n);

// Expected free energy G[k] = -sum_i q[i,k]*ln(q[i,k]+eps). Float32.
// q_outcomes is [n*K] row-major.
int metal_ai_expected_free_energy(
    const float* q_outcomes, float* out, int n, int K);

#ifdef __cplusplus
}
#endif

#endif /* METAL_ACTIVE_INFERENCE_H */
