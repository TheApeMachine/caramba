#ifndef XLA_EMBEDDING_H
#define XLA_EMBEDDING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque PJRT handles (redeclared to avoid including the full XLA header tree)
// ---------------------------------------------------------------------------

typedef struct PJRT_Client           PJRT_Client;
typedef struct PJRT_Buffer           PJRT_Buffer;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Error            PJRT_Error;

// ---------------------------------------------------------------------------
// High-level C wrappers implemented in embedding_xla.cc
// ---------------------------------------------------------------------------

// Initialize the PJRT client for the given platform ("cpu" or "gpu").
// Returns 0 on success, -1 on error.
int xla_embedding_init(const char* platform);

// Compile and cache the token-embedding executable for (n, d_model, vocab_size).
// Must be called after xla_embedding_init.
// Returns 0 on success, -1 on error.
int xla_compile_embedding(int n, int d_model, int vocab_size);

// Perform token embedding lookup.
//   tokens    — host double* of token IDs, length n
//   out       — caller-allocated host double*, length n * d_model
//   weight    — host double* weight table, length vocab_size * d_model
//   n         — number of tokens
//   d_model   — embedding dimension
//   vocab_size — vocabulary size
// Returns 0 on success, -1 on error.
int xla_token_embedding(
    const double* tokens,
    double*       out,
    const double* weight,
    int           n,
    int           d_model,
    int           vocab_size);

// Free all PJRT embedding resources.
void xla_embedding_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* XLA_EMBEDDING_H */
