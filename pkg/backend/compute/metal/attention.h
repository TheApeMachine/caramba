#ifndef METAL_ATTENTION_H
#define METAL_ATTENTION_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal pipelines for attention kernels.
// metallib_path: path to the compiled attention.metallib file.
// Returns 0 on success, -1 on failure.
int metal_init_attention(const char* metallib_path);

// Scaled Dot-Product Attention.
// q, k, v: float32 tensors of shape [batch*num_heads, seq_len, head_dim] (flattened)
// out:     float32 output tensor, same shape as q
// Returns 0 on success, -1 on failure.
int metal_sdpa(const float* q, const float* k, const float* v, float* out,
               int batch, int num_heads, int seq_len, int head_dim);

// Multi-Query Attention. K/V have 1 head; Q has num_heads heads.
// q:  [batch*num_heads, seq_len, head_dim]
// k,v:[batch*1,         seq_len, head_dim]
int metal_mqa(const float* q, const float* k, const float* v, float* out,
              int batch, int num_heads, int seq_len, int head_dim);

// Grouped Query Attention.
// q:  [batch*num_heads,    seq_len, head_dim]
// k,v:[batch*num_kv_heads, seq_len, head_dim]
int metal_gqa(const float* q, const float* k, const float* v, float* out,
              int batch, int num_heads, int num_kv_heads, int seq_len, int head_dim);

// Sliding Window Attention. window is the one-sided context radius.
int metal_sliding_window(const float* q, const float* k, const float* v, float* out,
                         int batch, int num_heads, int seq_len, int head_dim, int window);

#ifdef __cplusplus
}
#endif

#endif /* METAL_ATTENTION_H */
