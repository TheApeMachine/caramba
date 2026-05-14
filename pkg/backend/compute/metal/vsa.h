#ifndef METAL_VSA_H
#define METAL_VSA_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal device/queue and compile all VSA pipelines.
// metallib_path: absolute path to vsa.metallib (repo Makefile target).
// Returns 0 on success, -1 on failure.
// Pair metal_vsa_init with metal_vsa_cleanup before unload or process exit.
int metal_vsa_init(const char* metallib_path);

// Releases device, queue, and pipeline state created by metal_vsa_init (idempotent).
int metal_vsa_cleanup(void);

// Binding: out[i] = a[i] * b[i]  (Hadamard product, float32 on GPU)
int metal_vsa_bind(const float* a, const float* b, float* out, int n);

// L2-normalise: out[i] = in[i] / ||in||
int metal_vsa_l2normalize(const float* in, float* out, int n);

// Bundle count vectors laid out as count consecutive length-n vectors. The
// function computes out[i] = sum_j vectors[j*n + i], then L2-normalises that
// summed vector in out. vectors must contain count*n floats and out must have
// room for n floats.
int metal_vsa_bundle(const float* vectors, float* out, int count, int n);

// Dot product: out[0] = dot(a, b)  (cosine sim assuming unit-norm inputs)
int metal_vsa_dot(const float* a, const float* b, float* out, int n);

// Cyclic permutation: out[i] = src[(i-shift) mod n].
int metal_vsa_permute(const float* src, float* out, int n, int shift);

// Inverse cyclic permutation: out[i] = src[(i+shift) mod n].
int metal_vsa_inverse_permute(const float* src, float* out, int n, int shift);

// After a failed return from any metal_vsa_* call on this thread, inspect TLS diagnostics.
int metal_vsa_last_error_code(void);
const char* metal_vsa_last_error_message(void);

#ifdef __cplusplus
}
#endif

#endif /* METAL_VSA_H */
