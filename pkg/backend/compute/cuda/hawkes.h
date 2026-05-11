#ifndef CUDA_HAWKES_H
#define CUDA_HAWKES_H

#ifdef __cplusplus
extern "C" {
#endif

// Compute Hawkes process intensity for K processes.
// times[T], alpha[K], beta[K], mu[K], t (current time).
// out[K] = lambda values.
int cuda_hawkes_intensity(
    const double* times, const double* alpha,
    const double* beta,  const double* mu,
    double t,
    double* out,
    int K, int T
);

// Build excitation kernel matrix K[i,j] = alpha*exp(-beta*(t_j-t_i)) for j>i.
// times[T], out[T*T].
int cuda_hawkes_kernel_matrix(
    const double* times,
    double alpha, double beta,
    double* out,
    int T
);

// Log-likelihood: Σ log(lambda_i) - integral.
// intensities[T], integral scalar.
int cuda_hawkes_log_likelihood(
    const double* intensities,
    double integral,
    double* out,
    int T
);

// Simulate Hawkes process (thinning) for K independent processes.
// mu[K], alpha[K], beta[K], T_max, maxSteps per process.
// out[K*maxSteps] with sentinel -1 for unused slots.
int cuda_hawkes_simulate(
    const double* mu, const double* alpha,
    const double* beta,
    double T_max, int K, int maxSteps,
    double* out
);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_HAWKES_H */
