#ifndef CUDA_EMBEDDING_H
#define CUDA_EMBEDDING_H

#ifdef __cplusplus
extern "C" {
#endif

// cuda_token_embedding — token embedding lookup on CUDA.
//
//   tokens    — host pointer to double array of token IDs, length n
//   out       — caller-allocated host output, length n * d_model
//   weight    — host pointer to flat weight table, length vocab_size * d_model
//   n         — number of tokens (batch * seq_len)
//   d_model   — embedding dimension
//   vocab_size — vocabulary size
//
// Returns 0 on success, -1 on CUDA error.
int cuda_token_embedding(
    const double* tokens,
    double*       out,
    const double* weight,
    int           n,
    int           d_model,
    int           vocab_size);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_EMBEDDING_H */
