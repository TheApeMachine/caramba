#ifndef XLA_MASKING_H
#define XLA_MASKING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the XLA masking backend (reuses the same PJRT client as
// xla_init if called; otherwise call this standalone).
// Returns 0 on success, -1 on error.
int xla_masking_init(const char* platform);

// Generate causal mask: out[i*seq_len+j] = 0.0 if j<=i else -Inf
// n = seq_len (output is seq_len*seq_len doubles)
// Returns 0 on success, -1 on error.
int xla_causal_mask(double* out, int seq_len);

// Apply mask: out[i] = scores[i] + mask[i]
// n: total number of elements
// Returns 0 on success, -1 on error.
int xla_apply_mask(const double* scores, const double* mask, double* out, int n);

// Free masking resources.
void xla_masking_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* XLA_MASKING_H */
