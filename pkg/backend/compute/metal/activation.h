#ifndef METAL_ACTIVATION_H
#define METAL_ACTIVATION_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal device, command queue, and all compute pipelines.
// metallib_path: path to the compiled activation.metallib file.
// Returns 0 on success, -1 on failure.
int metal_init(const char* metallib_path);

// Compute ReLU: dst[i] = max(src[i], 0)
// n: number of elements
int metal_relu(const float* src, float* dst, int n);

// Compute Leaky ReLU: dst[i] = src[i] >= 0 ? src[i] : alpha * src[i]
int metal_leaky_relu(const float* src, float* dst, float alpha, int n);

// Compute GELU (tanh approximation)
int metal_gelu(const float* src, float* dst, int n);

// Compute element-wise tanh (rational approximation)
int metal_tanh(const float* src, float* dst, int n);

// Compute element-wise sigmoid
int metal_sigmoid(const float* src, float* dst, int n);

// Compute SwiGLU: dst[i] = sigmoid(src[i]) * src[n+i]
// src has 2*n elements (gates first, then values); dst has n elements.
int metal_swiglu(const float* src, float* dst, int n);

#ifdef __cplusplus
}
#endif

#endif /* METAL_ACTIVATION_H */
