#ifndef METAL_VSA_H
#define METAL_VSA_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal device/queue and compile all VSA pipelines.
// metallib_path: absolute path to vsa.metallib.
// Returns 0 on success, -1 on failure.
int metal_vsa_init(const char* metallib_path);

// Binding: out[i] = a[i] * b[i]  (Hadamard product, float32 on GPU)
int metal_vsa_bind(const float* a, const float* b, float* out, int n);

// L2-normalise: out[i] = in[i] / ||in||
int metal_vsa_l2normalize(const float* in, float* out, int n);

// Dot product: out[0] = dot(a, b)  (cosine sim assuming unit-norm inputs)
int metal_vsa_dot(const float* a, const float* b, float* out, int n);

#ifdef __cplusplus
}
#endif

#endif /* METAL_VSA_H */
