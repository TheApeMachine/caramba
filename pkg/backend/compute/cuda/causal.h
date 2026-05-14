#ifndef CUDA_CAUSAL_H
#define CUDA_CAUSAL_H

#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success, -1 on CUDA error.
// All src/dst are host pointers; device memory is managed internally.

// A^T @ B where A is [M x K] and B is [K x N] → C [M x N]
int cuda_causal_matmul_t(const double* A, const double* B, double* C, int M, int K, int N);

// A @ B where A is [M x K] and B is [K x N] → C [M x N]
int cuda_causal_matmul(const double* A, const double* B, double* C, int M, int K, int N);

// AXPY: dst[i] += scale * src[i], length n (device-side inputs/outputs)
int cuda_causal_axpy(double* dst, const double* src, double scale, int n);

// Parallel dot product reduction: out = sum(a[i]*b[i]), length n
int cuda_causal_dot(const double* a, const double* b, double* out, int n);

// Do-calculus: graph surgery on joint Gaussian
// cov [N*N], mask [N], values [N] → adjusted_mean [N] + adjusted_cov [N*N]
int cuda_causal_do_calculus(
    const double* cov, const double* mask, const double* values,
    double* out, int N
);

// Backdoor adjustment: OLS causal effect
// Y [T*ny], X [T*nx], Z [T*nz] → causal_effect [ny]
int cuda_causal_backdoor(
    const double* Y, const double* X, const double* Z,
    double* effect,
    int T, int ny, int nx, int nz
);

// IV estimate (2SLS)
// Z [T*nz], X [T*nx], Y [T*ny] → beta_iv [nx*ny]
int cuda_causal_iv(
    const double* Z, const double* X, const double* Y,
    double* beta_iv,
    int T, int nz, int nx, int ny
);

// CATE via outcome regression
// X [T*nx], treatment [T], Y [T] → cate [T]
int cuda_causal_cate(
    const double* X, const double* treatment, const double* Y,
    double* cate,
    int T, int nx
);

// DAG Markov factorization log probability
// X [T*N], adj [N*N] → log_prob [T]
int cuda_causal_dag_markov(
    const double* X, const double* adj,
    double* log_prob,
    int T, int N
);

// Counterfactual linear SCM: out[i*N_cf+j] = beta[i]*x_cf[j] + y_obs[i] - beta[i]*x_obs[i]
int cuda_causal_counterfactual(
    const double* x_obs, const double* y_obs, const double* beta, const double* x_cf,
    double* out,
    int N, int N_cf
);

// Frontdoor adjustment with equal-frequency binning on X and M.
int cuda_causal_frontdoor(
    const double* X, const double* M, const double* Y,
    double* effect,
    int T, int nx, int nm
);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_CAUSAL_H */
