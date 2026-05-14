#ifndef CUDA_TRAINING_H
#define CUDA_TRAINING_H

#ifdef __cplusplus
extern "C" {
#endif

int cuda_train_mse_loss(const double* predictions, const double* targets, double* out, int n);
int cuda_train_cross_entropy_loss(const double* logits, const double* targets, double* out, int n);
int cuda_train_mse_grad(const double* predictions, const double* targets, double* out, int n);
int cuda_train_cross_entropy_grad(const double* logits, const double* targets, double* out, int n);
int cuda_bench_accuracy(const double* predictions, const double* targets, double* out, int n);
int cuda_bench_f1_counts(const double* predictions, const double* targets, double* out, int n);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_TRAINING_H */
