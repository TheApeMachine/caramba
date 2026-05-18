package kernels

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
BF16 matmul parity. The kernel widens bf16→f32 in bulk via NEON,
runs a NEON axpy-based rank-1 update against an f32 accumulator, and
narrows back to bf16. The scalar reference performs the same widen
→ f32 mac → narrow, but lane-by-lane.

Both paths accumulate in f32 per §5.5. Their reduction order differs
(NEON walks a [M, K] × [K, N] tile producing accumulator updates in a
different order than the naive triple loop), so we allow a relative
tolerance scaled by inner dimension (which bounds the worst-case
error from associativity loss in summation).
*/

func TestBF16MatMulParity(t *testing.T) {
	cases := []struct{ M, K, N int }{
		{1, 8, 4},
		{4, 16, 8},
		{17, 32, 13},
		{64, 64, 64},
	}

	for _, kase := range cases {
		t.Run(fmt.Sprintf("M=%d_K=%d_N=%d", kase.M, kase.K, kase.N), func(t *testing.T) {
			rng := rand.New(rand.NewSource(0xB16 + int64(kase.M*1000+kase.K*100+kase.N)))

			lhs, _ := tensor.NewZeroed(mustShape([]int{kase.M, kase.K}), dtype.BFloat16)
			rhs, _ := tensor.NewZeroed(mustShape([]int{kase.K, kase.N}), dtype.BFloat16)
			out, _ := tensor.NewZeroed(mustShape([]int{kase.M, kase.N}), dtype.BFloat16)

			leftView, _ := lhs.BFloat16Native()
			rightView, _ := rhs.BFloat16Native()
			for index := range leftView {
				leftView[index] = dtype.NewBfloat16FromFloat32(float32(rng.NormFloat64()) * 0.5)
			}
			for index := range rightView {
				rightView[index] = dtype.NewBfloat16FromFloat32(float32(rng.NormFloat64()) * 0.5)
			}

			// Compute the literal scalar reference (triple-loop, f32 accumulation).
			scalarF32 := make([]float32, kase.M*kase.N)
			for row := 0; row < kase.M; row++ {
				for inner := 0; inner < kase.K; inner++ {
					l := (&leftView[row*kase.K+inner]).Float32()
					for col := 0; col < kase.N; col++ {
						r := (&rightView[inner*kase.N+col]).Float32()
						scalarF32[row*kase.N+col] += l * r
					}
				}
			}
			scalarBF16 := make([]dtype.BF16, kase.M*kase.N)
			for index, value := range scalarF32 {
				scalarBF16[index] = dtype.NewBfloat16FromFloat32(value)
			}

			if err := runMatMulBFloat16(lhs, rhs, out); err != nil {
				t.Fatal(err)
			}

			outView, _ := out.BFloat16Native()

			for index := range scalarBF16 {
				scalarValue := scalarBF16[index].Float32()
				neonValue := outView[index].Float32()

				// Allow inner-dimension scaled relative tolerance (bf16 has
				// ~7-bit mantissa = 2^-7 relative precision; with K
				// accumulations the bound grows as O(K * 2^-7) of the
				// answer magnitude, but the f32 accumulator absorbs most
				// of that; the final bf16 narrow is the dominant error).
				tolerance := float64(kase.K) * math.Max(math.Abs(float64(scalarValue)), 1.0) * 0x1p-7

				if math.Abs(float64(neonValue-scalarValue)) > tolerance {
					t.Fatalf("M=%d K=%d N=%d lane %d scalar=%g neon=%g tol=%g",
						kase.M, kase.K, kase.N, index, scalarValue, neonValue, tolerance,
					)
				}
			}
		})
	}
}

func BenchmarkBF16MatMul(b *testing.B) {
	for _, n := range []int{64, 256, 512} {
		n := n

		b.Run(fmt.Sprintf("%dx%dx%d", n, n, n), func(b *testing.B) {
			rng := rand.New(rand.NewSource(1))
			lhs, _ := tensor.NewZeroed(mustShape([]int{n, n}), dtype.BFloat16)
			rhs, _ := tensor.NewZeroed(mustShape([]int{n, n}), dtype.BFloat16)
			out, _ := tensor.NewZeroed(mustShape([]int{n, n}), dtype.BFloat16)

			leftView, _ := lhs.BFloat16Native()
			rightView, _ := rhs.BFloat16Native()
			for index := range leftView {
				leftView[index] = dtype.NewBfloat16FromFloat32(float32(rng.NormFloat64()))
			}
			for index := range rightView {
				rightView[index] = dtype.NewBfloat16FromFloat32(float32(rng.NormFloat64()))
			}

			// 2 * M * K * N flops.
			b.SetBytes(int64(2 * n * n * n))
			b.ResetTimer()

			for b.Loop() {
				_ = runMatMulBFloat16(lhs, rhs, out)
			}
		})
	}
}
