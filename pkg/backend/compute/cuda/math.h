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

// Scale: dst = src * (1/sqrt(dim))
int cuda_inv_sqrt_dim_scale(const double* src, double* dst, int n, int dim);

// Elementwise exp
int cuda_exp(const double* src, double* dst, int n);

// Elementwise log
int cuda_log(const double* src, double* dst, int n);

// Softmax over last dim. num_rows*dim_size elements.
int cuda_softmax(const double* src, double* dst, int num_rows, int dim_size);

// Layer norm. src/dst: num_rows*d_model. weight/bias: d_model.
int cuda_layernorm(const double* src, double* dst,
                   const double* weight, const double* bias,
                   int num_rows, int d_model, double eps);

// RMS norm. src/dst: num_rows*d_model. weight: d_model.
int cuda_rmsnorm(const double* src, double* dst,
                 const double* weight,
                 int num_rows, int d_model, double eps);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_MATH_H */
