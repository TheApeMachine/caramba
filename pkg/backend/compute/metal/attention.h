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
               int batch, int num_heads, int query_len, int key_value_len,
               int head_dim, int causal);

int metal_sdpa_tensor(const void* q, const void* k, const void* v, void* out,
                      int batch, int num_heads, int query_len, int key_value_len,
                      int key_value_stride, int head_dim, int causal);

int metal_kv_append_tensor(const void* previous_key, const void* previous_value,
                           const void* key_chunk, const void* value_chunk,
                           void* output_key, void* output_value,
                           int batch, int num_heads, int previous_len,
                           int chunk_len, int head_dim);

int metal_kv_repack_tensor(const void* previous_key, const void* previous_value,
                           void* output_key, void* output_value,
                           int batch, int num_heads, int current_len,
                           int head_dim, int previous_capacity,
                           int output_capacity);

int metal_kv_write_tensor(void* cache_key, void* cache_value,
                          const void* key_chunk, const void* value_chunk,
                          int batch, int num_heads, int start_len,
                          int chunk_len, int head_dim, int capacity);

// Multi-Query Attention. K/V have 1 head; Q has num_heads heads.
// q:  [batch*num_heads, seq_len, head_dim]
// k,v:[batch*1,         seq_len, head_dim]
int metal_mqa(const float* q, const float* k, const float* v, float* out,
              int batch, int num_heads, int seq_len, int head_dim);

int metal_mqa_tensor(const void* q, const void* k, const void* v, void* out,
                     int batch, int num_heads, int seq_len, int head_dim);

// Grouped Query Attention.
// q:  [batch*num_heads,    seq_len, head_dim]
// k,v:[batch*num_kv_heads, seq_len, head_dim]
int metal_gqa(const float* q, const float* k, const float* v, float* out,
              int batch, int num_heads, int num_kv_heads, int seq_len, int head_dim,
              int causal);

int metal_gqa_tensor(const void* q, const void* k, const void* v, void* out,
                     int batch, int num_heads, int num_kv_heads,
                     int query_len, int key_value_len, int key_value_stride,
                     int head_dim, int causal);

// Sliding Window Attention. window is the one-sided context radius.
int metal_sliding_window(const float* q, const float* k, const float* v, float* out,
                         int batch, int num_heads, int seq_len, int head_dim, int window);

int metal_sliding_window_tensor(const void* q, const void* k, const void* v, void* out,
                                int batch, int num_heads, int seq_len, int head_dim, int window);

#ifdef __cplusplus
}
#endif

#endif /* METAL_ATTENTION_H */
