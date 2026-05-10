#ifndef CUDA_SHAPE_H
#define CUDA_SHAPE_H

#ifdef __cplusplus
extern "C" {
#endif

// General N-D transpose (max rank 8) using double precision.
// shape  : array of rank dimension sizes (host pointer)
// rank   : number of dimensions
// dim0, dim1 : axes to swap
// n      : total number of elements
// Returns 0 on success, -1 on CUDA error.
int cuda_transpose(const double* src, double* dst,
                   const int* shape, int rank,
                   int dim0, int dim1, int n);

// Elementwise copy (reshape — data identical).
int cuda_copy(const double* src, double* dst, int n);

// Concatenate two flat buffers along a conceptual axis.
// n_a: elements in srcA; n_b: elements in srcB; dst must hold n_a+n_b.
int cuda_concat(const double* srcA, int n_a,
                const double* srcB, int n_b,
                double* dst);

// ViewAsHeads: [B,T,H,head_dim] -> [B,H,T,head_dim]
int cuda_view_as_heads(const double* src, double* dst,
                       int B, int T, int H, int head_dim);

// MergeHeads: [B,H,T,head_dim] -> [B,T,H,head_dim]
int cuda_merge_heads(const double* src, double* dst,
                     int B, int H, int T, int head_dim);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_SHAPE_H */
