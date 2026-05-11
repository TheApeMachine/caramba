#ifndef XLA_PREDICTIVE_CODING_H
#define XLA_PREDICTIVE_CODING_H

#ifdef __cplusplus
extern "C" {
#endif

int xla_pc_init(const char* platform);
void xla_pc_shutdown(void);

int xla_pc_prediction(const double* W, const double* r, double* dst, int D_out, int D_in);
int xla_pc_prediction_error(const double* x, const double* mu_hat,
                             const double* prec, double* dst, int n, int use_prec);
int xla_pc_update_representation(const double* r, const double* W,
                                  const double* eps_lower, const double* eps_self,
                                  double lr, double* dst, int D_out, int D_in);
int xla_pc_update_weights(const double* W, const double* eps, const double* r,
                           double lr, double* dst, int D_out, int D_in);

#ifdef __cplusplus
}
#endif

#endif /* XLA_PREDICTIVE_CODING_H */
