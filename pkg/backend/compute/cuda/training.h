#ifndef CUDA_TRAINING_H
#define CUDA_TRAINING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Computes mean squared error over n samples. predictions and targets are
// length-n float64 buffers; out is caller-allocated length 1. n must be > 0.
// Returns 0 on success and -1 for invalid arguments, allocation failure, or a
// CUDA runtime failure. Reentrant across host threads with normal CUDA context
// constraints.
int cuda_train_mse_loss(
    const double* predictions, const double* targets, double* out, size_t n);

// Computes mean cross entropy from raw logits and probability/one-hot targets.
// The implementation applies a max-shifted softmax before accumulating a single
// mean scalar in out[0]. logits and targets are length n in row-major flat
// layout for the caller's class grouping; n must be > 0.
// Returns 0 on success and -1 for invalid arguments, allocation failure, or a
// CUDA runtime failure.
int cuda_train_cross_entropy_loss(
    const double* logits, const double* targets, double* out, size_t n);

// Computes d(MSE)/d(predictions) into caller-allocated out[n]. n must be > 0.
// Returns 0 on success and -1 for invalid arguments, allocation failure, or a
// CUDA runtime failure.
int cuda_train_mse_grad(
    const double* predictions, const double* targets, double* out, size_t n);

// Computes softmax(logits)-targets into caller-allocated out[n]. Inputs are raw
// logits and probability/one-hot targets in the same flat layout as
// cuda_train_cross_entropy_loss; n must be > 0. Returns 0 on success and -1 on
// invalid arguments, allocation failure, or CUDA runtime failure.
int cuda_train_cross_entropy_grad(
    const double* logits, const double* targets, double* out, size_t n);

// Computes an overall argmax-style accuracy for n scores. predictions and
// targets are length-n score buffers; out must be caller-allocated length 1 and
// out[0] is 1 when the highest-scoring indices match, otherwise 0. n must be
// > 0. Returns 0 on success and -1 on invalid arguments or CUDA failure.
int cuda_metric_accuracy(const double* predictions, const double* targets, double* out, size_t n);

// Computes binary F1 confusion counts for n samples using threshold 0.5.
// predictions and targets are length-n buffers. out must be caller-allocated
// length 4 with layout TP, FP, FN, TN. n must be > 0. Returns 0 on success and
// -1 on invalid arguments or CUDA failure.
int cuda_metric_f1_counts(const double* predictions, const double* targets, double* out, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_TRAINING_H */
