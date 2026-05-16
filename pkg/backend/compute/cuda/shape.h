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
int cuda_transpose_device(const double* src, double* dst,
                          const int* shape, int rank,
                          int dim0, int dim1, int n);

// Elementwise copy (reshape — data identical).
int cuda_copy(const double* src, double* dst, int n);
int cuda_copy_device(const double* src, double* dst, int n);

// Concatenate two flat buffers along a conceptual axis.
// n_a: elements in srcA; n_b: elements in srcB; dst must hold n_a+n_b.
int cuda_concat(const double* srcA, int n_a,
                const double* srcB, int n_b,
                double* dst);
int cuda_concat_device(const double* srcA, int n_a,
                       const double* srcB, int n_b,
                       double* dst);

// Splits consecutive equal-sized ranges into groups along one logical dimension and
// writes the groups sequentially into dst. outer is the product of dimensions
// before the split axis, dim_size is the full size of that axis, split_size is
// the length of each chunk on that axis, and inner is the product of dimensions
// after the axis. dim_size must be divisible by split_size, yielding
// dim_size/split_size chunks. src and dst are row-major flat buffers of
// outer*dim_size*inner elements, and dst layout is
// [outer][chunk][offset_within_split_size][inner]. Returns 0 on success and -1 on invalid
// arguments or CUDA failure.
int cuda_split(const double* src, double* dst,
               int outer, int dim_size, int split_size, int inner);
int cuda_split_device(const double* src, double* dst,
                      int outer, int dim_size, int split_size, int inner);

// UpsampleNearest2D: [B,C,H,W] -> [B,C,H*scale_h,W*scale_w].
int cuda_upsample_nearest2d(const double* src, double* dst,
                            int B, int C, int H, int W,
                            int scale_h, int scale_w);
int cuda_upsample_nearest2d_device(const double* src, double* dst,
                                   int B, int C, int H, int W,
                                   int scale_h, int scale_w);

// ViewAsHeads: [B,T,H,head_dim] -> [B,H,T,head_dim]
int cuda_view_as_heads(const double* src, double* dst,
                       int B, int T, int H, int head_dim);
int cuda_view_as_heads_device(const double* src, double* dst,
                              int B, int T, int H, int head_dim);

// MergeHeads: [B,H,T,head_dim] -> [B,T,H,head_dim]
int cuda_merge_heads(const double* src, double* dst,
                     int B, int H, int T, int head_dim);
int cuda_merge_heads_device(const double* src, double* dst,
                            int B, int H, int T, int head_dim);

// LastToken: [outer, seq_len, feature] -> [outer, feature].
int cuda_last_token(const double* src, double* dst,
                    int outer, int seq_len, int feature);
int cuda_last_token_device(const double* src, double* dst,
                           int outer, int seq_len, int feature);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_SHAPE_H */
