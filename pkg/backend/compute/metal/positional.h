#ifndef METAL_POSITIONAL_H
#define METAL_POSITIONAL_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal pipelines for positional kernels.
// metallib_path: path to compiled positional.metallib.
// Returns 0 on success, -1 on failure.
int metal_positional_init(const char* metallib_path);

// RoPE: apply rotary position embeddings.
// x, out: float arrays of length total_heads * seq_len * head_dim
// cos_table, sin_table: float arrays of length seq_len * (head_dim/2)
// total_heads = batch * num_heads
int metal_rope(
    const float* x,
    float*       out,
    const float* cos_table,
    const float* sin_table,
    int          seq_len,
    int          head_dim,
    int          rope_mode,
    int          total_heads);

int metal_rope_tensor(
    const void*  x,
    void*        out,
    const float* cos_table,
    const float* sin_table,
    int          seq_len,
    int          head_dim,
    int          rope_mode,
    int          total_heads);

// ALiBi: compute attention bias tensor.
// out: float array of length num_heads * seq_len_q * seq_len_k
// slopes: float array of length num_heads
int metal_alibi(
    float*       out,
    const float* slopes,
    int          num_heads,
    int          seq_len_q,
    int          seq_len_k);

#ifdef __cplusplus
}
#endif

#endif /* METAL_POSITIONAL_H */
