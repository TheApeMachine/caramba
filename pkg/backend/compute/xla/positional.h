#ifndef XLA_POSITIONAL_H
#define XLA_POSITIONAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the PJRT client for positional ops.
// Must be called once before any positional dispatch.
// Returns 0 on success, -1 on error.
int xla_positional_init(const char* platform);

// Compile executables for the given tensor dimensions.
// Must be called after xla_positional_init.
// Returns 0 on success, -1 on error.
int xla_compile_positional(int total_heads, int seq_len, int head_dim,
                            int num_heads_alibi, int seq_len_q, int seq_len_k);

// RoPE: apply rotary position embeddings.
// x, out: host double arrays of length total_heads * seq_len * head_dim
// cos_table, sin_table: host double arrays of length seq_len * (head_dim/2)
// Returns 0 on success, -1 on error.
int xla_rope(
    const double* x,
    double*       out,
    const double* cos_table,
    const double* sin_table,
    int           seq_len,
    int           head_dim,
    int           total_heads);

// ALiBi: compute attention bias tensor.
// out: host double array of length num_heads * seq_len_q * seq_len_k
// slopes: host double array of length num_heads
// causal: 1 = signed (k-q), 0 = absolute value
int xla_alibi(
    double*       out,
    const double* slopes,
    int           num_heads,
    int           seq_len_q,
    int           seq_len_k,
    int           causal);

// Free all PJRT resources.
void xla_positional_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* XLA_POSITIONAL_H */
