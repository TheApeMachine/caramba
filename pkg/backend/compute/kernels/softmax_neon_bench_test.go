package kernels_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func BenchmarkSoftmaxFloat32(b *testing.B) {
	for _, n := range []int{1024, 4096} {
		n := n
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			shape, _ := tensor.NewShape([]int{n})
			in, _ := tensor.NewZeroed(shape, dtype.Float32)
			out, _ := tensor.NewZeroed(shape, dtype.Float32)
			inView, _ := in.Float32Native()
			rng := rand.New(rand.NewSource(1))
			for i := range inView {
				inView[i] = float32(rng.NormFloat64() * 3)
			}

			kernel, _ := kernels.Default.Lookup("softmax", kernels.Signature{
				Layout:  tensor.LayoutDense,
				Inputs:  []dtype.DType{dtype.Float32},
				Outputs: []dtype.DType{dtype.Float32},
			})

			b.SetBytes(int64(n * 4 * 2))
			b.ResetTimer()
			for b.Loop() {
				_ = kernel.Run(in, out)
			}
		})
	}
}
