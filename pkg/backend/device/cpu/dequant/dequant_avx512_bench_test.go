//go:build amd64

package dequant

import (
	"fmt"
	"math/rand"
	"testing"
)

func BenchmarkDequantInt8AVX512(b *testing.B) {
	for _, length := range []int{64, 1024, 8192, 65536} {
		length := length

		b.Run(fmt.Sprintf("N=%d", length), func(b *testing.B) {
			source := make([]int8, length)
			rng := rand.New(rand.NewSource(1))

			for index := range source {
				source[index] = int8(rng.Intn(256) - 128)
			}

			destination := make([]float32, length)

			b.SetBytes(int64(length * 5))
			b.ResetTimer()

			for b.Loop() {
				dequantInt8AVX512(destination, source, 0.0875, -13)
			}
		})
	}
}

func BenchmarkDequantInt4AVX512(b *testing.B) {
	for _, length := range []int{64, 1024, 8192, 65536} {
		length := length

		b.Run(fmt.Sprintf("N=%d", length), func(b *testing.B) {
			bytes := int4BytesFromLength(length, 1)
			destination := make([]float32, length)

			b.SetBytes(int64(length / 2))
			b.ResetTimer()

			for b.Loop() {
				dequantInt4AVX512(destination, bytes, length, 0.0625, 3)
			}
		})
	}
}
