// Package normalization implements GroupNorm, InstanceNorm, and
// BatchNorm (eval mode) for float32, bfloat16, and float16 tensors.
//
// Float32 row statistics and apply kernels follow the pick-at-init model via
// select_{amd64,other}.go on amd64 (AVX-512 when available).
package normalization
