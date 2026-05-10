#ifndef XLA_ACTIVATION_H
#define XLA_ACTIVATION_H

// Minimal PJRT C API surface needed by the XLA activation backend.
// The full PJRT C API is defined in xla/pjrt/c/pjrt_c_api.h from the
// openxla/xla repository.  We redeclare only the types and function
// pointer signatures we use here so the Go CGo layer does not need to
// depend on the full XLA header tree.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

typedef struct PJRT_Client        PJRT_Client;
typedef struct PJRT_Buffer        PJRT_Buffer;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Error         PJRT_Error;

// ---------------------------------------------------------------------------
// High-level C wrappers implemented in activation_xla.cc
// ---------------------------------------------------------------------------

// Initialize the PJRT client for the given platform ("cpu" or "gpu").
// Returns 0 on success, -1 on error.
int xla_init(const char* platform);

// Compile and cache all activation executables.  Must be called after
// xla_init and before any dispatch function.
// Returns 0 on success, -1 on error.
int xla_compile_activations(int n_elements);

// Dispatch functions.  All take host double* pointers; the XLA layer
// manages device transfers internally.
// n is the number of *output* elements.
// Returns 0 on success, -1 on error.

int xla_relu(const double* src, double* dst, int n);
int xla_leaky_relu(const double* src, double* dst, double alpha, int n);
int xla_gelu(const double* src, double* dst, int n);
int xla_tanh_act(const double* src, double* dst, int n);
int xla_sigmoid(const double* src, double* dst, int n);
// src has 2*n elements (gates then values); dst has n elements.
int xla_swiglu(const double* src, double* dst, int n);

// Free all PJRT resources.
void xla_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* XLA_ACTIVATION_H */
