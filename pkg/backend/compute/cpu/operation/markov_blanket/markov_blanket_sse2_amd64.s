#include "textflag.h"

// This file intentionally empty — SSE2 kernels are defined in markov_blanket_avx2_amd64.s
// (same package, same translation unit, Plan 9 assembler merges them).
// The TEXT directives for matvecSSE2 and subVecSSE2 reside there to avoid
// duplicate symbol errors while keeping the two sets of implementations
// separated conceptually.
