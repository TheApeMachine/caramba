//go:build amd64

package matmul

import (
	"fmt"
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkMatMulF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	for _, size := range []int{64, 256, 512} {
		b.Run(fmt.Sprintf("%dx%dx%d", size, size, size), func(b *testing.B) {
			rows := size
			inner := size
			cols := size

			left := randomMatmulFloat32Slice(rows*inner, 1)
			right := randomMatmulFloat32Slice(inner*cols, 2)
			out := make([]float32, rows*cols)

			b.SetBytes(int64(2 * rows * inner * cols))
			b.ResetTimer()

			for b.Loop() {
				MatmulFloat32AVX512(out, left, right, rows, inner, cols)
			}
		})
	}
}

func BenchmarkMatMulF32Generic(b *testing.B) {
	for _, size := range []int{64, 256, 512} {
		b.Run(fmt.Sprintf("%dx%dx%d", size, size, size), func(b *testing.B) {
			rows := size
			inner := size
			cols := size

			left := randomMatmulFloat32Slice(rows*inner, 1)
			right := randomMatmulFloat32Slice(inner*cols, 2)
			out := make([]float32, rows*cols)

			b.SetBytes(int64(2 * rows * inner * cols))
			b.ResetTimer()

			for b.Loop() {
				matmulFloat32Scalar(out, left, right, rows, inner, cols)
			}
		})
	}
}
