//go:build amd64

package quant

import (
	"fmt"
	"math/rand"
	"testing"
)

func BenchmarkQuantInt8AVX512(b *testing.B) {
	for _, length := range []int{64, 1024, 8192, 65536} {
		length := length

		b.Run(fmt.Sprintf("N=%d", length), func(b *testing.B) {
			source := make([]float32, length)
			rng := rand.New(rand.NewSource(1))

			for index := range source {
				source[index] = float32(rng.NormFloat64()) * 10
			}

			destination := make([]int8, length)

			b.SetBytes(int64(length * 5))
			b.ResetTimer()

			for b.Loop() {
				quantInt8AVX512(destination, source, 0.125, 7)
			}
		})
	}
}
