package kernels_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func BenchmarkMatMulF32NEONTile(b *testing.B) {
	for _, n := range []int{64, 256, 512} {
		n := n
		b.Run(fmt.Sprintf("%dx%dx%d", n, n, n), func(b *testing.B) {
			shape, _ := tensor.NewShape([]int{n, n})
			a, _ := tensor.NewZeroed(shape, dtype.Float32)
			c, _ := tensor.NewZeroed(shape, dtype.Float32)
			d, _ := tensor.NewZeroed(shape, dtype.Float32)
			aView, _ := a.Float32Native()
			cView, _ := c.Float32Native()
			rng := rand.New(rand.NewSource(1))
			for i := range aView {
				aView[i] = float32(rng.NormFloat64())
				cView[i] = float32(rng.NormFloat64())
			}

			kernel, _ := kernels.Default.Lookup("matmul", kernels.Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
				Outputs: []dtype.DType{dtype.Float32},
			})

			b.SetBytes(int64(2 * n * n * n))
			b.ResetTimer()
			for b.Loop() {
				_ = kernel.Run(a, c, d)
			}
		})
	}
}
