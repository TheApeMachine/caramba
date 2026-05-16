#ifndef XLA_ATTENTION_H
#define XLA_ATTENTION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque handle types (same as activation.h — may be included together)
// ---------------------------------------------------------------------------

#ifndef PJRT_HANDLE_TYPES_DEFINED
#define PJRT_HANDLE_TYPES_DEFINED
typedef struct PJRT_Client           PJRT_Client;
typedef struct PJRT_Buffer           PJRT_Buffer;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Error            PJRT_Error;
#endif

// ---------------------------------------------------------------------------
// High-level C wrappers for attention operations, implemented in
// attention_xla.cc.  All functions take host double* pointers; the XLA layer
// manages device transfers internally.
// Returns 0 on success, -1 on error.
// ---------------------------------------------------------------------------

// Scaled Dot-Product Attention.
// q, k, v: flattened [batch, num_heads, seq_len, head_dim] (row-major, double)
// out:     same shape as q
int xla_sdpa(const double* q, const double* k, const double* v, double* out,
             int batch, int num_heads, int seq_len, int head_dim);

// Multi-Query Attention.  K/V have 1 head per batch item.
// q:  [batch, num_heads, seq_len, head_dim]
// k,v:[batch, 1,         seq_len, head_dim]
int xla_mqa(const double* q, const double* k, const double* v, double* out,
            int batch, int num_heads, int seq_len, int head_dim);

// Grouped Query Attention.
// q:  [batch, num_heads,    seq_len, head_dim]
// k,v:[batch, num_kv_heads, seq_len, head_dim]
int xla_gqa(const double* q, const double* k, const double* v, double* out,
            int batch, int num_heads, int num_kv_heads, int seq_len, int head_dim,
            int causal);

// Sliding Window Attention.
int xla_sliding_window(const double* q, const double* k, const double* v, double* out,
                       int batch, int num_heads, int seq_len, int head_dim, int window);

#ifdef __cplusplus
}
#endif

#endif /* XLA_ATTENTION_H */
