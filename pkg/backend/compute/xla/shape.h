#ifndef XLA_SHAPE_H
#define XLA_SHAPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types (shared with activation.h — redeclare guard).
#ifndef XLA_OPAQUE_TYPES_DEFINED
#define XLA_OPAQUE_TYPES_DEFINED
typedef struct PJRT_Client           PJRT_Client;
typedef struct PJRT_Buffer           PJRT_Buffer;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Error            PJRT_Error;
#endif

// ---------------------------------------------------------------------------
// High-level C wrappers implemented in shape_xla.cc
//
// All functions assume xla_init() (from activation.h) has already been called
// to create the global PJRT client.  Executables are compiled and cached
// per-call.
// ---------------------------------------------------------------------------

// Transpose two dimensions of a rank-R tensor stored row-major.
// shape  : R dimension sizes
// rank   : number of dimensions
// dim0, dim1 : axes to swap
// n      : total number of elements
// Returns 0 on success, -1 on error.
int xla_transpose(const double* src, double* dst,
                  const int* shape, int rank,
                  int dim0, int dim1, int n);

// Elementwise copy (reshape — data unchanged).
int xla_copy(const double* src, double* dst, int n);

// Concatenate along the last axis represented as two flat buffers.
// n_a elements from srcA, n_b from srcB -> n_a+n_b in dst.
int xla_concat(const double* srcA, int n_a,
               const double* srcB, int n_b,
               double* dst);

// ViewAsHeads: [B,T,H,head_dim] -> [B,H,T,head_dim]
int xla_view_as_heads(const double* src, double* dst,
                      int B, int T, int H, int head_dim);

// MergeHeads: [B,H,T,head_dim] -> [B,T,H,head_dim]
int xla_merge_heads(const double* src, double* dst,
                    int B, int H, int T, int head_dim);

#ifdef __cplusplus
}
#endif

#endif /* XLA_SHAPE_H */
