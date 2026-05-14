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

// Cyclic permutation by shift positions. Positive shift rotates values to the
// right: src[i] is written to out[(i+shift) mod n]. Negative shifts rotate in
// the opposite direction and all shifts are normalized modulo n. |shift| may be
// larger than n; it is normalized modulo n before applying the rotation. shift==0
// copies src to out. n must be > 0; invalid arguments return -1. src and out
// must not alias because the implementation is not in-place safe.
int cuda_vsa_permute(const double* src, double* out, int n, int shift);

// Inverse cyclic permutation by shift positions. Applying
// cuda_vsa_inverse_permute with the same shift value after cuda_vsa_permute
// using separate src and out buffers restores the original array. Positive shift
// uses the same direction definition as cuda_vsa_permute and is converted to the
// opposite normalized rotation internally without overflowing INT_MIN. n must
// be > 0, |shift| may exceed n, and invalid arguments return -1. src and out
// must not alias.
int cuda_vsa_inverse_permute(const double* src, double* out, int n, int shift);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_VSA_H */
