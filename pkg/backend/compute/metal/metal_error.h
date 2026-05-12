#ifndef METAL_ERROR_H
#define METAL_ERROR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define METAL_OK 0

/*
Return-code vocabulary (documented in one place). Values are **not** interchangeable
across every entrypoint — e.g. active inference uses -2 for “not initialised” while
causal uses -2 for “graph / mask error”. Go wrappers surface these as opaque integers.

Active inference / predictive coding style:
  0   OK
 -1   invalid argument
 -2   not initialised
 -3   numeric failure (non-finite, bad conditioning where applicable)

Causal (metal_causal_*):
  0   OK
 -1   invalid argument
 -2   graph / mask error (e.g. DAG cycle)
 -3   not initialised
 -4   allocation failure
 -5   numeric (Cholesky / OLS / inversion)

Hawkes: same integer pattern as active inference (-3 means not init in hawkes .m).

metal_error_string maps any of the above to a short label for logging.
*/
const char *metal_error_string(int code);

#ifdef __cplusplus
}
#endif

#endif /* METAL_ERROR_H */
