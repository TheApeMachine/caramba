#ifndef METAL_PREDICTIVE_CODING_H
#define METAL_PREDICTIVE_CODING_H

#ifdef __cplusplus
extern "C" {
#endif

int metal_pc_init(const char* metallib_path);

// Top-down prediction: dst[D_out] = W[D_out*D_in] @ r[D_in]
int metal_pc_prediction(const float* W, const float* r, float* dst, int D_out, int D_in);

// Precision-weighted prediction error: dst = prec * (x - mu_hat); pass NULL prec to skip.
int metal_pc_prediction_error(const float* x, const float* mu_hat,
                               const float* prec, float* dst, int n, int use_prec);

// Representation update: dst[D_in] = r + lr*(W^T @ eps_lower - eps_self)
int metal_pc_update_representation(const float* r, const float* W,
                                    const float* eps_lower, const float* eps_self,
                                    float lr, float* dst, int D_out, int D_in);

// Weight update: dst[D_out*D_in] = W + lr * eps ⊗ r
int metal_pc_update_weights(const float* W, const float* eps, const float* r,
                             float lr, float* dst, int D_out, int D_in);

#ifdef __cplusplus
}
#endif

#endif /* METAL_PREDICTIVE_CODING_H */
