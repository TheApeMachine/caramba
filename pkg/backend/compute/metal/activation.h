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

// Compute tanh-form GELU
int metal_gelu(const float* src, float* dst, int n);

// Compute element-wise tanh
int metal_tanh(const float* src, float* dst, int n);

// Compute element-wise sigmoid
int metal_sigmoid(const float* src, float* dst, int n);

// Compute Swish: dst[i] = src[i] * sigmoid(src[i])
int metal_swish(const float* src, float* dst, int n);

// Compute SELU: scale*x for positives, scale*alpha*(exp(x)-1) otherwise.
int metal_selu(const float* src, float* dst, int n);

// Compute SwiGLU across one row: dst[i] = src[i] * sigmoid(src[i]) * src[n+i].
// src has 2*n elements (gates first, then values); dst has n elements.
int metal_swiglu(const float* src, float* dst, int n);

/*
Device-resident variants: src/dst are MTLBuffer pointers from metal_tensor_*.
Each kernel reads n float elements from src (or 2*n for SwiGLU), writes n floats to dst.
metal_leaky_relu_tensor passes alpha as the kernel scalar parameter.
Returns 0 on success, -1 on invalid arguments or GPU encode/submit failure.
*/
int metal_relu_tensor(const void* src, void* dst, int n);
int metal_leaky_relu_tensor(const void* src, void* dst, float alpha, int n);
int metal_gelu_tensor(const void* src, void* dst, int n);
int metal_tanh_tensor(const void* src, void* dst, int n);
int metal_sigmoid_tensor(const void* src, void* dst, int n);
int metal_swish_tensor(const void* src, void* dst, int n);
int metal_selu_tensor(const void* src, void* dst, int n);

/*
metal_swiglu_tensor applies SwiGLU row-wise over the final tensor dimension.
input_width is the source final dimension, split as [gate | value] per row.
dst holds n floats. Requires n > 0 and an even input_width.
*/
int metal_swiglu_tensor(const void* src, void* dst, int n, int input_width);

#ifdef __cplusplus
}
#endif

#endif /* METAL_ACTIVATION_H */
