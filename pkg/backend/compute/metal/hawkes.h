#ifndef METAL_HAWKES_H
#define METAL_HAWKES_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal pipelines for Hawkes process operations.
int metal_hawkes_init(const char* metallib_path);

// Intensity: lambda[K] = mu[K] + alpha[K] * sum_i exp(-beta[K]*(t-times[i]))
int metal_hawkes_intensity(
    const float* times, const float* alpha,
    const float* beta,  const float* mu,
    float t,
    float* out,
    int K, int T
);

// Kernel matrix: out[T*T], K[i,j]=alpha*exp(-beta*(t_j-t_i)) for j>i.
int metal_hawkes_kernel_matrix(
    const float* times,
    float alpha, float beta,
    float* out,
    int T
);

// Log-likelihood: sum log(lambda_i) - integral. Returns scalar into out[1].
int metal_hawkes_log_likelihood(
    const float* intensities,
    float integral,
    float* out,
    int T
);

// Simulate (Ogata thinning). out[K*maxSteps], sentinel -1.
int metal_hawkes_simulate(
    const float* mu, const float* alpha,
    const float* beta,
    float T_max, int K, int maxSteps,
    float* out
);

#ifdef __cplusplus
}
#endif

#endif /* METAL_HAWKES_H */
