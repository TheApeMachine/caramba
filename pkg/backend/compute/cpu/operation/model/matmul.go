package model

import cpumath "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"

/*
MatMulFn computes C = A · B where A is [M×K] and B is [K×N], both stored
row-major in flat slices. Returns C as a flat [M×N] slice.
Requires M,K,N > 0 and len(a) >= M*K, len(b) >= K*N.

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
	if M <= 0 || K <= 0 || N <= 0 {
		panic("model: CPUMatMul requires M, K, N > 0")
	}

	if len(a) < M*K || len(b) < K*N {
		panic("model: CPUMatMul slice lengths too short for given dimensions")
	}

	c := make([]float64, M*N)
	cpumath.MatMul(c, a, b, M, K, N)

	return c
}

/*
CPUMatMulInto writes C = A · B into the provided dst slice (avoids allocation).
Requires len(dst) >= M*N, len(a) >= M*K, len(b) >= K*N.
*/
func CPUMatMulInto(dst, a, b []float64, M, K, N int) {
	if len(dst) < M*N {
		panic("model: CPUMatMulInto: dst too short")
	}

	cpumath.MatMul(dst, a, b, M, K, N)
}
