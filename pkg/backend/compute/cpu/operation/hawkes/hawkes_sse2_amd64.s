#include "textflag.h"

// SSE2 kernels are defined in hawkes_avx2_amd64.s to avoid duplicate symbols.
// This file is intentionally a stub so the build system sees both _avx2 and _sse2
// files per the project convention.
