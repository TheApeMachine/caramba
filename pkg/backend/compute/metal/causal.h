#ifndef METAL_CAUSAL_H
#define METAL_CAUSAL_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal causal pipelines.
// metallib_path: absolute path to a compiled .metallib containing causal kernels.
// Returns 0 on success, -1 on failure.
int metal_causal_init(const char* metallib_path);

// Elementwise AXPY: dst[i] += scale * src[i], float32, length n.
int metal_causal_axpy(float* dst, const float* src, float scale, int n);

// Dot product: returns sum(a[i]*b[i]) via Metal reduction, length n.
int metal_causal_dot(const float* a, const float* b, float* out, int n);

// Elementwise subtraction: dst[i] = a[i] - b[i], length n.
int metal_causal_sub(float* dst, const float* a, const float* b, int n);

// Matrix-vector product: dst[rows] = W[rows x cols] @ x[cols], float32.
int metal_causal_matvec(float* dst, const float* W, const float* x, int rows, int cols);

// Do-calculus: graph surgery on joint Gaussian.
// cov [N*N], mask [N], values [N] → out [N + N*N] (adjusted_mean ++ adjusted_cov).
int metal_causal_do_calculus(
    const float* cov, const float* mask, const float* values,
    float* out, int N
);

// Backdoor adjustment causal effect.
// Y [T*ny], X [T*nx], Z [T*nz] → effect [ny].
int metal_causal_backdoor(
    const float* Y, const float* X, const float* Z,
    float* effect,
    int T, int ny, int nx, int nz
);

// IV estimate (2SLS): Z [T*nz], X [T*nx], Y [T*ny] → beta_iv [nx*ny].
int metal_causal_iv(
    const float* Z, const float* X, const float* Y,
    float* beta_iv,
    int T, int nz, int nx, int ny
);

// CATE via outcome regression: X [T*nx], treatment [T], Y [T] → cate [T].
int metal_causal_cate(
    const float* X, const float* treatment, const float* Y,
    float* cate,
    int T, int nx
);

// DAG Markov factorization log probability: X [T*N], adj [N*N] → log_prob [T].
int metal_causal_dag_markov(
    const float* X, const float* adj,
    float* log_prob,
    int T, int N
);

#ifdef __cplusplus
}
#endif

#endif /* METAL_CAUSAL_H */
