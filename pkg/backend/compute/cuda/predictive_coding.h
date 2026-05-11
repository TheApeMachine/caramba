#ifndef CUDA_PREDICTIVE_CODING_H
#define CUDA_PREDICTIVE_CODING_H

#ifdef __cplusplus
extern "C" {
#endif

// Top-down prediction: dst[D_out] = W[D_out*D_in] @ r[D_in]
int cuda_pc_prediction(const double* W, const double* r, double* dst, int D_out, int D_in);

// Precision-weighted prediction error: dst[N] = precision[N] * (x[N] - mu_hat[N])
// Pass NULL for precision to skip weighting.
int cuda_pc_prediction_error(const double* x, const double* mu_hat,
                              const double* precision, double* dst, int n);

// Representation update: r_new[D_in] = r[D_in] + lr * (W^T @ eps_lower[D_out] - eps_self[D_in])
int cuda_pc_update_representation(const double* r, const double* W,
                                   const double* eps_lower, const double* eps_self,
                                   double lr, double* dst, int D_out, int D_in);

// Weight update: W_new[D_out*D_in] = W + lr * eps[D_out] ⊗ r[D_in]
int cuda_pc_update_weights(const double* W, const double* eps, const double* r,
                            double lr, double* dst, int D_out, int D_in);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_PREDICTIVE_CODING_H */
