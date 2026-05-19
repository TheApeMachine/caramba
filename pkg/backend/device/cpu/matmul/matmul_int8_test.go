package matmul

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
INT8 matmul parity. NEON path uses SDOT for the inner dot; scalar
reference uses naive int32(a)*int32(b) accumulation. Both compute
exact integer arithmetic with int32 accumulators, so the result
should be bit-identical.
*/

func TestInt8MatMulParity(t *testing.T) {
	cases := []struct{ M, K, N int }{
		{1, 16, 4},
		{4, 32, 8},
		{17, 48, 13},
		{64, 64, 64},
		{31, 17, 23},
	}

	for _, kase := range cases {
		t.Run(fmt.Sprintf("M=%d_K=%d_N=%d", kase.M, kase.K, kase.N), func(t *testing.T) {
			rng := rand.New(rand.NewSource(0x18 + int64(kase.M*1000+kase.K*100+kase.N)))

			lhs, _ := tensor.NewZeroed(mustShape([]int{kase.M, kase.K}), dtype.Int8)
			rhs, _ := tensor.NewZeroed(mustShape([]int{kase.K, kase.N}), dtype.Int8)
			out, _ := tensor.NewZeroed(mustShape([]int{kase.M, kase.N}), dtype.Int32)

			leftView, _ := lhs.Int8Native()
			rightView, _ := rhs.Int8Native()
			for index := range leftView {
				leftView[index] = int8(rng.Intn(256) - 128)
			}
			for index := range rightView {
				rightView[index] = int8(rng.Intn(256) - 128)
			}

			scalar := make([]int32, kase.M*kase.N)
			for row := 0; row < kase.M; row++ {
				for col := 0; col < kase.N; col++ {
					var sum int32
					for k := 0; k < kase.K; k++ {
						sum += int32(leftView[row*kase.K+k]) * int32(rightView[k*kase.N+col])
					}
					scalar[row*kase.N+col] = sum
				}
			}

			if err := RunMatMulInt8(lhs, rhs, out); err != nil {
				t.Fatal(err)
			}

			outView, _ := out.Int32Native()

			for index := range scalar {
				if scalar[index] != outView[index] {
					t.Fatalf("M=%d K=%d N=%d lane %d scalar=%d neon=%d",
						kase.M, kase.K, kase.N, index, scalar[index], outView[index],
					)
				}
			}
		})
	}
}

func BenchmarkInt8MatMul(b *testing.B) {
	for _, n := range []int{64, 256, 512} {
		n := n

		b.Run(fmt.Sprintf("%dx%dx%d", n, n, n), func(b *testing.B) {
			rng := rand.New(rand.NewSource(1))
			lhs, _ := tensor.NewZeroed(mustShape([]int{n, n}), dtype.Int8)
			rhs, _ := tensor.NewZeroed(mustShape([]int{n, n}), dtype.Int8)
			out, _ := tensor.NewZeroed(mustShape([]int{n, n}), dtype.Int32)

			leftView, _ := lhs.Int8Native()
			rightView, _ := rhs.Int8Native()
			for index := range leftView {
				leftView[index] = int8(rng.Intn(256) - 128)
			}
			for index := range rightView {
				rightView[index] = int8(rng.Intn(256) - 128)
			}

			b.SetBytes(int64(2 * n * n * n))
			b.ResetTimer()

			for b.Loop() {
				_ = RunMatMulInt8(lhs, rhs, out)
			}
		})
	}
}
