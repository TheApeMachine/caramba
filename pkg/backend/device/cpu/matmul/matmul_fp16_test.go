package matmul

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestFP16MatMulParity(t *testing.T) {
	cases := []struct{ M, K, N int }{
		{1, 8, 4},
		{4, 16, 8},
		{17, 32, 13},
		{64, 64, 64},
	}

	for _, kase := range cases {
		t.Run(fmt.Sprintf("M=%d_K=%d_N=%d", kase.M, kase.K, kase.N), func(t *testing.T) {
			rng := rand.New(rand.NewSource(0xF16 + int64(kase.M*1000+kase.K*100+kase.N)))

			lhs, _ := tensor.NewZeroed(mustShape([]int{kase.M, kase.K}), dtype.Float16)
			rhs, _ := tensor.NewZeroed(mustShape([]int{kase.K, kase.N}), dtype.Float16)
			out, _ := tensor.NewZeroed(mustShape([]int{kase.M, kase.N}), dtype.Float16)

			leftView, _ := lhs.Float16Native()
			rightView, _ := rhs.Float16Native()
			for index := range leftView {
				leftView[index] = dtype.Fromfloat32(float32(rng.NormFloat64()) * 0.5)
			}
			for index := range rightView {
				rightView[index] = dtype.Fromfloat32(float32(rng.NormFloat64()) * 0.5)
			}

			scalarF32 := make([]float32, kase.M*kase.N)
			for row := 0; row < kase.M; row++ {
				for inner := 0; inner < kase.K; inner++ {
					l := leftView[row*kase.K+inner].Float32()
					for col := 0; col < kase.N; col++ {
						r := rightView[inner*kase.N+col].Float32()
						scalarF32[row*kase.N+col] += l * r
					}
				}
			}
			scalarF16 := make([]dtype.F16, kase.M*kase.N)
			for index, value := range scalarF32 {
				scalarF16[index] = dtype.Fromfloat32(value)
			}

			if err := runMatMulFloat16(lhs, rhs, out); err != nil {
				t.Fatal(err)
			}

			outView, _ := out.Float16Native()

			for index := range scalarF16 {
				scalarValue := scalarF16[index].Float32()
				neonValue := outView[index].Float32()

				// fp16 has 10-bit mantissa = ~2^-10 relative precision.
				tolerance := float64(kase.K) * math.Max(math.Abs(float64(scalarValue)), 1.0) * 0x1p-10

				if math.Abs(float64(neonValue-scalarValue)) > tolerance {
					t.Fatalf("M=%d K=%d N=%d lane %d scalar=%g neon=%g tol=%g",
						kase.M, kase.K, kase.N, index, scalarValue, neonValue, tolerance,
					)
				}
			}
		})
	}
}

func BenchmarkFP16MatMul(b *testing.B) {
	for _, n := range []int{64, 256, 512} {
		n := n

		b.Run(fmt.Sprintf("%dx%dx%d", n, n, n), func(b *testing.B) {
			rng := rand.New(rand.NewSource(1))
			lhs, _ := tensor.NewZeroed(mustShape([]int{n, n}), dtype.Float16)
			rhs, _ := tensor.NewZeroed(mustShape([]int{n, n}), dtype.Float16)
			out, _ := tensor.NewZeroed(mustShape([]int{n, n}), dtype.Float16)

			leftView, _ := lhs.Float16Native()
			rightView, _ := rhs.Float16Native()
			for index := range leftView {
				leftView[index] = dtype.Fromfloat32(float32(rng.NormFloat64()))
			}
			for index := range rightView {
				rightView[index] = dtype.Fromfloat32(float32(rng.NormFloat64()))
			}

			b.SetBytes(int64(2 * n * n * n))
			b.ResetTimer()

			for b.Loop() {
				_ = runMatMulFloat16(lhs, rhs, out)
			}
		})
	}
}
