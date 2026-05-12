#ifndef METAL_PREDICTIVE_CODING_H
#define METAL_PREDICTIVE_CODING_H

#ifdef __cplusplus
extern "C" {
#endif

/*
Return codes: 0 = success; -1 invalid args; -3 not initialised; -4/-5 internal.
metal_pc_init loads predictive_coding.metallib (see repo Makefile).
The Objective-C bridge uses a serial dispatch queue; Go wraps with a mutex.
Call metal_pc_init before other functions; metal_pc_shutdown pairs with init.
*/

int metal_pc_init(const char *metallib_path);

int metal_pc_shutdown(void);

int metal_pc_prediction(const float *W, const float *r, float *dst, int D_out, int D_in);

/* If prec is NULL, dst[i] = x[i]-mu_hat[i]; else dst[i] = prec[i]*(x[i]-mu_hat[i]). */
int metal_pc_prediction_error(
    const float *x, const float *mu_hat,
    const float *prec, float *dst, int n);

int metal_pc_update_representation(
    const float *r, const float *W,
    const float *eps_lower, const float *eps_self,
    float lr, float *dst, int D_out, int D_in);

int metal_pc_update_weights(
    const float *W, const float *eps, const float *r,
    float lr, float *dst, int D_out, int D_in);

#ifdef __cplusplus
}
#endif

#endif /* METAL_PREDICTIVE_CODING_H */
