#ifndef PKG_BACKEND_COMPUTE_XLA_XLA_VSA_H
#define PKG_BACKEND_COMPUTE_XLA_XLA_VSA_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * xla_vsa_init / xla_vsa_shutdown:
 * platform: NUL-terminated PJRT plugin selector (e.g. "cpu" or "gpu").
 * xla_vsa_init returns 0 on success, -1 on plugin-load / client-create failure.
 * Pair each successful init with a shutdown. Concurrent init/shutdown vs ops
 * is undefined unless externally serialised.
 */
int  xla_vsa_init(const char* platform);
void xla_vsa_shutdown(void);

/**
 * xla_vsa_get_last_error: NUL-terminated message from the last PJRT error
 * handled in this process (empty string if none). Pointer is valid until the
 * next VSA call that may mutate the buffer.
 */
const char* xla_vsa_get_last_error(void);

/**
 * xla_vsa_bind: elementwise multiply (Hadamard product / VSA binding).
 * a, b: host double arrays of length n. out: pre-allocated host array of length n.
 * Returns 0 on success, -1 on XLA/PJRT error or non-positive n.
 */
int xla_vsa_bind(
    const double* a, const double* b, double* out, int n);

/**
 * xla_vsa_bundle: sum num_vecs input vectors then L2-normalise.
 * vecs: array of num_vecs host double pointers, each of length n.
 * out: pre-allocated host array of length n.
 * Returns 0 on success, -1 on error.
 */
int xla_vsa_bundle(
    const double** vecs, int num_vecs, double* out, int n);

/**
 * xla_vsa_similarity: dot product of two unit-norm VSA hypervectors.
 * a, b: host double arrays of length n. out: pre-allocated host array of length 1.
 * Returns 0 on success, -1 on error.
 */
int xla_vsa_similarity(
    const double* a, const double* b, double* out, int n);

/**
 * xla_vsa_permute performs a circular shift of src into out.
 * @param src Host array of length n.
 * @param out Pre-allocated host array of length n.
 * @param n Vector length; must be positive.
 * @param shift Signed circular offset. Negative values and |shift| >= n are
 * normalized modulo n.
 * @return 0 on success, -1 for null pointers, non-positive n, or PJRT failure.
 */
int xla_vsa_permute(
    const double* src, double* out, int n, int shift);

/**
 * xla_vsa_inverse_permute applies the inverse circular shift for shift.
 * @param src Host array of length n.
 * @param out Pre-allocated host array of length n.
 * @param n Vector length; must be positive.
 * @param shift Signed circular offset used by xla_vsa_permute; it is normalized
 * modulo n before inversion.
 * @return 0 on success, -1 for null pointers, non-positive n, or PJRT failure.
 */
int xla_vsa_inverse_permute(
    const double* src, double* out, int n, int shift);

#ifdef __cplusplus
}
#endif

#endif /* PKG_BACKEND_COMPUTE_XLA_XLA_VSA_H */
