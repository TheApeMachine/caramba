#ifndef CUDA_VSA_H
#define CUDA_VSA_H

#ifdef __cplusplus
extern "C" {
#endif

// VSA (Vector Symbolic Algebra) CUDA kernels — FHRR style (real-valued, float64).
// All host pointers; device memory is managed internally.
// Returns 0 on success, -1 on CUDA error.

// Binding: out[i] = a[i] * b[i]  (Hadamard product)
int cuda_vsa_bind(const double* a, const double* b, double* out, int n);

// Bundling: out = L2-normalise(sum of num_vecs input vectors, each length n)
int cuda_vsa_bundle(const double** vecs, int num_vecs, double* out, int n);

// Similarity: out[0] = dot(a, b)  (cosine sim assuming unit-norm inputs)
int cuda_vsa_similarity(const double* a, const double* b, double* out, int n);

// Cyclic permutation by shift positions.
int cuda_vsa_permute(const double* src, double* out, int n, int shift);

// Inverse cyclic permutation by shift positions.
int cuda_vsa_inverse_permute(const double* src, double* out, int n, int shift);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_VSA_H */
