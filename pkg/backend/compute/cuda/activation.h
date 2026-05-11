#ifndef CUDA_ACTIVATION_H
#define CUDA_ACTIVATION_H

#ifdef __cplusplus
extern "C" {
#endif

// All functions:
//   src   — host pointer to input doubles
//   dst   — host pointer to output doubles (caller-allocated)
//   n     — number of output elements
//   Returns 0 on success, -1 on CUDA error.
// Device memory is managed internally.

int cuda_relu(const double* src, double* dst, int n);
int cuda_leaky_relu(const double* src, double* dst, double alpha, int n);
int cuda_gelu(const double* src, double* dst, int n);
int cuda_tanh(const double* src, double* dst, int n);
int cuda_sigmoid(const double* src, double* dst, int n);

// src has 2*n doubles (gates then values); dst has n doubles.
int cuda_swiglu(const double* src, double* dst, int n);

// Device-resident variants. src/dst are CUDA device pointers.
int cuda_relu_device(const void* src, void* dst, int n);
int cuda_leaky_relu_device(const void* src, void* dst, double alpha, int n);
int cuda_gelu_device(const void* src, void* dst, int n);
int cuda_tanh_device(const void* src, void* dst, int n);
int cuda_sigmoid_device(const void* src, void* dst, int n);
int cuda_swiglu_device(const void* src, void* dst, int n);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_ACTIVATION_H */
