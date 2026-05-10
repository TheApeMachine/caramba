#ifndef METAL_SHAPE_H
#define METAL_SHAPE_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal device, command queue, and shape compute pipelines.
// metallib_path: path to the compiled shape.metallib file.
// Returns 0 on success, -1 on failure.
int metal_shape_init(const char* metallib_path);

// Transpose two dimensions of an N-D tensor stored row-major (float32).
// src    : input flat buffer (n elements)
// dst    : output flat buffer (n elements, caller-allocated)
// shape  : dimension sizes (rank ints)
// rank   : number of dimensions
// dim0   : first dimension to swap
// dim1   : second dimension to swap
// n      : total number of elements
int metal_transpose(const float* src, float* dst,
                    const int* shape, int rank,
                    int dim0, int dim1, int n);

// Copy src to dst (reshape — data identical, only logical shape differs).
int metal_copy(const float* src, float* dst, int n);

// Concatenate two flat buffers into dst.
// n_a: number of elements in srcA; n_b in srcB; dst must hold n_a+n_b elements.
int metal_concat(const float* srcA, int n_a,
                 const float* srcB, int n_b,
                 float* dst);

// ViewAsHeads: [B,T,H,head_dim] -> [B,H,T,head_dim]
int metal_view_as_heads(const float* src, float* dst,
                        int B, int T, int H, int head_dim);

// MergeHeads: [B,H,T,head_dim] -> [B,T,H,head_dim]
int metal_merge_heads(const float* src, float* dst,
                      int B, int H, int T, int head_dim);

#ifdef __cplusplus
}
#endif

#endif /* METAL_SHAPE_H */
