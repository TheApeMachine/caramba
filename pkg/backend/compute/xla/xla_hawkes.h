#ifndef XLA_HAWKES_H
#define XLA_HAWKES_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize XLA PJRT client for Hawkes process operations.
int xla_hawkes_init(const char* platform);

// Intensity: out[K] = mu[K] + alpha[K] * sum_i exp(-beta[K]*(t-times[i]))
int xla_hawkes_intensity(
    const double* times, const double* alpha,
    const double* beta,  const double* mu,
    double t,
    double* out,
    int K, int T
);

// Kernel matrix: out[T*T].
int xla_hawkes_kernel_matrix(
    const double* times,
    double alpha, double beta,
    double* out,
    int T
);

// Log-likelihood scalar.
int xla_hawkes_log_likelihood(
    const double* intensities,
    double integral,
    double* out,
    int T
);

// Simulate: out[K*maxSteps], sentinel -1.
int xla_hawkes_simulate(
    const double* mu, const double* alpha,
    const double* beta,
    double T_max, int K, int maxSteps,
    double* out
);

void xla_hawkes_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* XLA_HAWKES_H */
