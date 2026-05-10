#ifndef METAL_EMBEDDING_H
#define METAL_EMBEDDING_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal device, command queue, and the token_embedding pipeline.
// metallib_path: absolute path to embedding.metallib compiled from embedding.metal.
// Returns 0 on success, -1 on failure.
int metal_embedding_init(const char* metallib_path);

// Perform token embedding lookup.
//   tokens    — float32 array of token IDs, length batch_seq
//   out       — caller-allocated float32 output, length batch_seq * d_model
//   weight    — flat weight table, length vocab_size * d_model
//   batch_seq — batch * seq_len (total number of tokens)
//   d_model   — embedding dimension
//   vocab_size — vocabulary size (unused at kernel level, provided for validation)
// Returns 0 on success, -1 on failure.
int metal_token_embedding(
    const float* tokens,
    float*       out,
    const float* weight,
    int          batch_seq,
    int          d_model,
    int          vocab_size);

#ifdef __cplusplus
}
#endif

#endif /* METAL_EMBEDDING_H */
