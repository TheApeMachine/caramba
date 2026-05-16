#ifndef CUDA_POSITIONAL_H
#define CUDA_POSITIONAL_H

#ifdef __cplusplus
extern "C" {
#endif

// RoPE: apply rotary position embeddings.
// x, out: host double arrays of length total_heads * seq_len * head_dim
// cos_table, sin_table: host double arrays of length seq_len * (head_dim/2)
// total_heads = batch * num_heads
// Returns 0 on success, -1 on CUDA error.
int cuda_rope(
    const double* x,
    double*       out,
    const double* cos_table,
    const double* sin_table,
    int           seq_len,
    int           head_dim,
    int           rope_mode,
    int           total_heads);

// ALiBi: compute attention bias tensor.
// out: host double array of length num_heads * seq_len_q * seq_len_k
// slopes: host double array of length num_heads
// causal: 1 = signed (k-q), 0 = absolute value
int cuda_alibi(
    double*       out,
    const double* slopes,
    int           num_heads,
    int           seq_len_q,
    int           seq_len_k,
    int           causal);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_POSITIONAL_H */
