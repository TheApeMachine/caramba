#include "textflag.h"
// SSE2 primitives are all implemented in primitives_avx2_amd64.s as TEXT symbols
// with SSE2 in the name. This file is intentionally empty — all SSE2 TEXT
// symbols live in the AVX2 file to avoid duplicate symbol errors.
