#ifndef METAL_CAUSAL_H
#define METAL_CAUSAL_H

#ifdef __cplusplus
extern "C" {
#endif

/*
Metal implementation (float32; linear algebra runs on the GPU via causal.metallib).

Return codes: 0 success; -1 invalid argument; -2 graph / mask error (e.g. DAG cycle);
 -3 not initialised; -4 allocation failure; -5 numeric (inversion / OLS failure).

metal_causal_init loads causal.metallib (Makefile); metal_causal_shutdown clears state (idempotent).
Not safe for concurrent use without external synchronisation.

Float buffers are host pointers; lengths are element counts; matrices are row-major.

Elementwise / matvec helpers:
  metal_causal_axpy: dst[i] += scale*src[i]
  metal_causal_dot: writes dot(a,b) into *out
  metal_causal_sub: dst[i] = a[i] - b[i]
  metal_causal_matvec: dst[rows] = W[rows×cols row-major] @ x[cols]
  W[row*cols + col] is row row, column col, x indexed 0..cols-1.

metal_causal_do_calculus: cov is N×N row-major joint covariance; mask[i]!=0 marks
  do-intervention on variable i; values[i] is the intervention level. out is
  N + N*N floats: adjusted mean[0..N-1] then adjusted covariance row-major.

metal_causal_backdoor: ATE via outcome regression (reference CPU). Y is T×ny row-major,
  X is T×nx, Z is T×nz; effect length ny (float32). No normalisation in-kernel.

metal_causal_iv: 2SLS. Z [T×nz], X [T×nx], Y [T×ny] row-major. beta_iv row-major
  [nx×ny]: coefficient for X_i on outcome j at beta_iv[i*ny + j].

metal_causal_cate: X [T×nx], treatment [T], Y [T]; output cate [T].

metal_causal_dag_markov: X [T×N] row-major (sample t, variable n: X[t*N+n]); adj
  [N×N] row-major, adj[i*N+j]!=0 means directed edge parent j → child i; must be
  a DAG. log_prob length T.
*/

int metal_causal_init(const char *metallib_path);

int metal_causal_shutdown(void);

int metal_causal_axpy(float *dst, const float *src, float scale, int n);

int metal_causal_dot(const float *a, const float *b, float *out, int n);

int metal_causal_sub(float *dst, const float *a, const float *b, int n);

int metal_causal_matvec(float *dst, const float *W, const float *x, int rows, int cols);

int metal_causal_do_calculus(
    const float *cov, const float *mask, const float *values,
    float *out, int N);

int metal_causal_backdoor(
    const float *Y, const float *X, const float *Z,
    float *effect,
    int T, int ny, int nx, int nz);

int metal_causal_iv(
    const float *Z, const float *X, const float *Y,
    float *beta_iv,
    int T, int nz, int nx, int ny);

int metal_causal_cate(
    const float *X, const float *treatment, const float *Y,
    float *cate,
    int T, int nx);

int metal_causal_dag_markov(
    const float *X, const float *adj,
    float *log_prob,
    int T, int N);

#ifdef __cplusplus
}
#endif

#endif /* METAL_CAUSAL_H */
