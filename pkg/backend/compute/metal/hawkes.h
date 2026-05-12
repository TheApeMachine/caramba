#ifndef METAL_HAWKES_H
#define METAL_HAWKES_H

#ifdef __cplusplus
extern "C" {
#endif

/*
API: functions return 0 on success, negative on error (-1 invalid, -3 not initialised).
Thread-safety: not safe for concurrent calls without external synchronisation.

metal_hawkes_init loads hawkes.metallib (see repo Makefile).
Callers allocate output buffers; kernels do not take ownership.

metal_hawkes_intensity: out length K, one intensity per dimension at time t.
metal_hawkes_kernel_matrix: full T×T row-major K[i,j] at out[i*T+j]; upper triangle
  j>i has alpha*exp(-beta*(t_j-t_i)), else 0 (including diagonal).
metal_hawkes_log_likelihood: writes scalar sum_i log(intensities[i]) - integral into out[0];
  requires out pointing to at least one float; intensities length T.
metal_hawkes_simulate: out layout K blocks of maxSteps floats: out[d*maxSteps + i] is
  event time i for dimension d; unused entries are -1.0f sentinel. Ogata thinning;
  returns 0 on success (truncation policy: silently capped at maxSteps per dim).
*/

int metal_hawkes_init(const char *metallib_path);

int metal_hawkes_cleanup(void);

int metal_hawkes_intensity(
    const float *times, const float *alpha,
    const float *beta, const float *mu,
    float t,
    float *out,
    int K, int T);

int metal_hawkes_kernel_matrix(
    const float *times,
    float alpha, float beta,
    float *out,
    int T);

int metal_hawkes_log_likelihood(
    const float *intensities,
    float integral,
    float *out,
    int T);

int metal_hawkes_simulate(
    const float *mu, const float *alpha,
    const float *beta,
    float T_max, int K, int maxSteps,
    float *out);

#ifdef __cplusplus
}
#endif

#endif /* METAL_HAWKES_H */
