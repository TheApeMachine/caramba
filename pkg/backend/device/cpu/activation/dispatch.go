// Package activation implements dense element-wise activations, gated linear
// units, and row softmax for float32, float16, and bfloat16.
//
// Float32 kernels follow the pospop dispatch model: each ISA ships a complete
// assembly implementation (AVX-512, AVX2, SSE2 on amd64/386, NEON on arm64).
// At init time the highest-tier available implementation is selected per
// operation. Float16 and bfloat16 use precomputed LUTs filled from the scalar
// reference in pkg/backend/device/cpu/math.
package activation
