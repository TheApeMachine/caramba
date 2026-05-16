#ifndef METAL_MARKOV_BLANKET_H
#define METAL_MARKOV_BLANKET_H

#ifdef __cplusplus
extern "C" {
#endif

/*
Return codes: 0 success; negative errors (invalid args, not init, I/O, numeric).
Thread-safety: not safe for concurrent use without external locking.

metal_mb_init loads markov_blanket.metallib (see repo Makefile).
Callers allocate output buffers; kernels do not take ownership.
*/

int metal_mb_init(const char *metallib_path);

int metal_mb_cleanup(void);

/*
Partition: x length N; masks = [smask|amask|imask|emask] each length N float32.
out length Ns+Na+Ni+Ne. Returns non-zero if mask overlaps (more than one 1 per index).
*/
int metal_mb_partition(
    const float *x, const float *masks,
    float *out,
    int N, int Ns, int Na, int Ni, int Ne);
int metal_mb_partition_tensor(
    const void *x, const void *masks,
    void *out,
    int N, int Ns, int Na, int Ni, int Ne);

/*
Internal flow: W row-major Ni×Ns; x_sens length Ns; bias length Ni; out length Ni.
out = W @ x_sens + bias.
*/
int metal_mb_flow_internal(
    const float *x_sens, const float *W, const float *bias,
    float *out,
    int Ni, int Ns);
int metal_mb_flow_internal_tensor(
    const void *x_sens, const void *W, const void *bias,
    void *out,
    int Ni, int Ns);

/*
Active flow: W row-major Na×Ni; x_int length Ni; bias length Na; out length Na.
out = W @ x_int + bias (same pattern as flow_internal; see metal_mb_flow_internal).
*/
int metal_mb_flow_active(
    const float *x_int, const float *W, const float *bias,
    float *out,
    int Na, int Ni);
int metal_mb_flow_active_tensor(
    const void *x_int, const void *W, const void *bias,
    void *out,
    int Na, int Ni);

/*
Gaussian MI approximation: MI = 0.5*(logdet Sigma_X + logdet Sigma_Y - logdet Sigma_joint).
X row-major T×N (sample t at row t: X[t*N+n]); Y row-major T×M. out[0] receives scalar MI.
Requires T>=2, N>0, M>0.
*/
int metal_mb_mutual_information(
    const float *X, const float *Y,
    float *out,
    int T, int N, int M);
int metal_mb_mutual_information_tensor(
    const void *X, const void *Y,
    void *out,
    int T, int N, int M);

#ifdef __cplusplus
}
#endif

#endif /* METAL_MARKOV_BLANKET_H */
