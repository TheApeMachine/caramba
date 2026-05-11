#ifndef PKG_BACKEND_COMPUTE_XLA_XLA_CAUSAL_H
#define PKG_BACKEND_COMPUTE_XLA_XLA_CAUSAL_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * xla_causal_init(const char* platform): initialise PJRT causal kernels.
 * Returns 0 on success, -1 on failure.
 */
int xla_causal_init(const char* platform);

/**
 * xla_causal_shutdown: release PJRT causal resources.
 */
void xla_causal_shutdown(void);

/**
 * xla_causal_do_calculus: graph-surgery posterior under joint Gaussian.
 * cov [N*N], mask [N], values [N] → out [N + N*N].
 * Returns 0 on success, -1 on error.
 */
int xla_causal_do_calculus(
    const double* cov, const double* mask, const double* values,
    double* out, int N
);

/**
 * xla_causal_backdoor: backdoor-adjusted causal effect.
 * Y [T*ny], X [T*nx], Z [T*nz] → effect [ny].
 */
int xla_causal_backdoor(
    const double* Y, const double* X, const double* Z,
    double* effect,
    int T, int ny, int nx, int nz
);

/**
 * xla_causal_frontdoor: frontdoor-adjusted causal effect.
 * X [T], M [T], Y [T] → effect [nx] where nx = N_x bins.
 */
int xla_causal_frontdoor(
    const double* X, const double* M, const double* Y,
    double* effect,
    int T, int nx, int nm
);

/**
 * xla_causal_counterfactual: abduction-action-prediction.
 * X_obs [N], Y_obs [N], beta [N], X_cf [N_cf] → Y_cf [N_cf].
 */
int xla_causal_counterfactual(
    const double* X_obs, const double* Y_obs,
    const double* beta, const double* X_cf,
    double* Y_cf,
    int N, int N_cf
);

/**
 * xla_causal_iv: two-stage least squares.
 * Z [T*nz], X [T*nx], Y [T*ny] → beta_iv [nx*ny].
 */
int xla_causal_iv(
    const double* Z, const double* X, const double* Y,
    double* beta_iv,
    int T, int nz, int nx, int ny
);

/**
 * xla_causal_cate: conditional average treatment effect.
 * X [T*nx], treatment [T], Y [T] → cate [T].
 */
int xla_causal_cate(
    const double* X, const double* treatment, const double* Y,
    double* cate,
    int T, int nx
);

/**
 * xla_causal_dag_markov: Markov factorization log probability.
 * X [T*N], adj [N*N] → log_prob [T].
 */
int xla_causal_dag_markov(
    const double* X, const double* adj,
    double* log_prob,
    int T, int N
);

#ifdef __cplusplus
}
#endif

#endif /* PKG_BACKEND_COMPUTE_XLA_XLA_CAUSAL_H */
