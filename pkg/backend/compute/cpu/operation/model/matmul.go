package model

import cpumath "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"

/*
MatMulFn computes C = A · B where A is [M×K] and B is [K×N], both stored
row-major in flat slices. Returns C as a flat [M×N] slice.

This is the injection point for accelerated backends. Metal, CUDA, and XLA
supply their own via NewLoRA / NewAdapter constructors. The CPU path
dispatches to AVX2/NEON/SSE2 via the math package's SIMD kernels.
*/
type MatMulFn func(a, b []float64, M, K, N int) []float64

/*
CPUMatMul dispatches to the AVX2/NEON/SSE2 matmul kernel in the math package.
Used by default when no accelerated backend is injected.
*/
func CPUMatMul(a, b []float64, M, K, N int) []float64 {
	c := make([]float64, M*N)
	cpumath.MatMul(c, a, b, M, K, N)

	return c
}
