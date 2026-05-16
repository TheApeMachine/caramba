#ifndef METAL_ACTIVE_INFERENCE_H
#define METAL_ACTIVE_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

/*
API conventions: return 0 on success; negative error codes otherwise.
See `metal_error_string` in `metal_error.c` for short labels.

metal_ai_init loads active_inference.metallib from the path you pass (absolute path recommended).

Thread-safety: the Objective-C bridge uses a private serial queue; still pair init/cleanup
explicitly. Go callers should use one ActiveInferenceOps per logical device and rely on its mutex.

metal_ai_init must succeed before other entrypoints. metal_ai_cleanup releases
reference state and is idempotent; after cleanup, init must be called again
before compute functions run.

Pairing: always call metal_ai_cleanup once per successful init cycle before
unload or exit.
*/

int metal_ai_init(const char *metallib_path);

/* Release state from metal_ai_init (idempotent). Returns 0 on success. */
int metal_ai_cleanup(void);

/*
Variational free energy F = 0.5*sum(mu_i^2 + exp(log_sigma_i) - log_sigma_i - 1).
mu, log_sigma: host float32, n elements each, non-NULL when n>0.
out: non-NULL; receives scalar F in out[0]. n==0 writes 0 to out[0].
Returns 0 on success, non-zero on error.
*/
int metal_ai_free_energy(const float *mu, const float *log_sigma, float *out, int n);

/*
Belief update (same formulas as CPU reference):
  out[0..n-1]   = mu - lr*(mu + pred_err)
  out[n..2n-1]  = log_sigma - lr*(exp(log_sigma) - 1)
out must hold 2*n float32 values.
*/
int metal_ai_belief_update(
    const float *mu, const float *log_sigma,
    const float *pred_err, float lr,
    float *out, int n);

/*
Precision-weighted error: out[i] = err[i] * exp(clamp(log_prec[i], -80, 80)).
err, log_prec, out: length n float32 arrays. Returns -3 if non-finite inputs.
*/
int metal_ai_precision_weight(
    const float *err, const float *log_prec, float *out, int n);

/*
Expected free energy: G[k] = -sum_i clamp(q[i,k],0,1)*ln(clamp(q[i,k],0,1)+eps).
q_outcomes: row-major n×K (index i*K+k). out: length K. eps must be finite and >0.
*/
int metal_ai_expected_free_energy(
    const float *q_outcomes, float *out, int n, int K, float eps);

#ifdef __cplusplus
}
#endif

#endif /* METAL_ACTIVE_INFERENCE_H */
