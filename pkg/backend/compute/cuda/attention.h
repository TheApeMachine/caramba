#ifndef CUDA_ATTENTION_H
#define CUDA_ATTENTION_H

#ifdef __cplusplus
extern "C" {
#endif

// All functions take host double* pointers; device memory is managed internally.
// Returns 0 on success, -1 on CUDA error.

// Scaled Dot-Product Attention.
// q, k, v: [batch*num_heads, seq_len, head_dim] (row-major, double)
// out:     same shape as q
int cuda_sdpa(const double* q, const double* k, const double* v, double* out,
              int batch, int num_heads, int seq_len, int head_dim);

// Multi-Query Attention. K/V have 1 head per batch item.
// q:  [batch*num_heads, seq_len, head_dim]
// k,v:[batch*1,         seq_len, head_dim]
int cuda_mqa(const double* q, const double* k, const double* v, double* out,
             int batch, int num_heads, int seq_len, int head_dim);

// Grouped Query Attention.
// q:  [batch*num_heads,    seq_len, head_dim]
// k,v:[batch*num_kv_heads, seq_len, head_dim]
int cuda_gqa(const double* q, const double* k, const double* v, double* out,
             int batch, int num_heads, int num_kv_heads, int seq_len, int head_dim,
             int causal);

// Sliding Window Attention. window is the one-sided context radius.
int cuda_sliding_window(const double* q, const double* k, const double* v, double* out,
                        int batch, int num_heads, int seq_len, int head_dim, int window);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_ATTENTION_H */
