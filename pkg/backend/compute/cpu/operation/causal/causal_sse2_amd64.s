#include "textflag.h"

// causal_sse2_amd64.s — SSE2-named TEXT symbols (·matVecSSE2, ·axpySSE2, etc.) are
// implemented in causal_avx2_amd64.s so the linker sees exactly one definition per symbol.
// causal_amd64.go dispatches to AVX2 or SSE2 routines based on runtime CPU feature flags.
