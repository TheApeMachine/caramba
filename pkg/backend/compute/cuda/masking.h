#ifndef CUDA_MASKING_H
#define CUDA_MASKING_H

#ifdef __cplusplus
extern "C" {
#endif

// cuda_causal_mask: fills out[i*seq_len+j] = 0.0 if j<=i else -Inf
// out: caller-allocated, n = seq_len*seq_len doubles
// Returns 0 on success, -1 on CUDA error.
int cuda_causal_mask(double* out, int seq_len);

// cuda_apply_mask: out[i] = scores[i] + mask[i]
// n: total number of elements
// Returns 0 on success, -1 on CUDA error.
int cuda_apply_mask(const double* scores, const double* mask, double* out, int n);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_MASKING_H */
