#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#ifdef __cplusplus
extern "C" {
#endif

// All double-precision math kernels.
// src/dst are host pointers; device memory is managed internally.
// Returns 0 on success, -1 on CUDA error.

// Matrix multiply: C[M*N] = A[M*K] x B[K*N]
int cuda_matmul(const double* A, const double* B, double* C, int M, int K, int N);

// Elementwise: out = a + b
int cuda_add(const double* a, const double* b, double* out, int n);

// Elementwise: out = a * b
int cuda_mul(const double* a, const double* b, double* out, int n);

// Device-resident variants (CUDA device pointers to float64 data).
// Returns 0 on success, -1 on CUDA error.

int cuda_matmul_device(const double* A, const double* B, double* C, int M, int K, int N);
int cuda_add_device(const double* a, const double* b, double* out, int n);
int cuda_mul_device(const double* a, const double* b, double* out, int n);

// Fused matmul + bias (+ optional GELU). bias_n may be N (broadcast along rows)
// or M*N (full matrix bias). apply_gelu: non-zero applies gelu_device to each output.
// sync_device: non-zero synchronizes the device before return (default); pass 0 to overlap launches (caller must sync later).
int cuda_matmul_add_device(
    const double* A, const double* B, const double* bias, double* C,
    int M, int K, int N, int bias_n, int apply_gelu, int sync_device
);

// Scale: dst = src * (1/sqrt(dim))
int cuda_inv_sqrt_dim_scale(const double* src, double* dst, int n, int dim);

// Elementwise exp
int cuda_exp(const double* src, double* dst, int n);

// Elementwise log
int cuda_log(const double* src, double* dst, int n);

// Softmax over last dim. num_rows*dim_size elements.
int cuda_softmax(const double* src, double* dst, int num_rows, int dim_size);

// LogSumExp over last dim. src has num_rows*dim_size elements; dst has num_rows.
int cuda_logsumexp(const double* src, double* dst, int num_rows, int dim_size);

// Dropout. If training is zero or p is zero, copies input to output.
int cuda_dropout(const double* src, double* dst, int n, double p, int training, int seed);

// Layer norm. src/dst: num_rows*d_model. weight/bias: d_model.
int cuda_layernorm(const double* src, double* dst,
                   const double* weight, const double* bias,
                   int num_rows, int d_model, double eps);

// RMS norm. src/dst: num_rows*d_model. weight: d_model.
int cuda_rmsnorm(const double* src, double* dst,
                 const double* weight,
                 int num_rows, int d_model, double eps);

// Group norm over NCHW tensors. weight/bias length is channels.
int cuda_groupnorm(const double* src, double* dst,
                   const double* weight, const double* bias,
                   int batch, int channels, int height, int width,
                   int groups, double eps);

// Elementwise sign: out[i] = +1 if src[i] > 0, -1 if < 0, 0 if == 0.
int cuda_sign(const double* src, double* dst, int n);

// Outer product: dst[i*N+j] = a[i] * b[j]. M=len(a), N=len(b).
int cuda_outer(const double* a, const double* b, double* dst, int M, int N);

// Optimizer primitives
int cuda_axpy(double* dst, const double* src, double scale, int n);
int cuda_scale(double* dst, double s, int n);
int cuda_sqrt_vec(const double* src, double* dst, int n);
int cuda_add_scalar(double* dst, double scalar, int n);
int cuda_div_vec(const double* a, const double* b, double* dst, int n);
int cuda_clamp_vec(double* dst, double lo, double hi, int n);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_MATH_H */
