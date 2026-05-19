//go:build arm64

package neon

import (
	"fmt"
	"math"
	"testing"
)

func TestAbsFloat32NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomFloat32Slice(n, 0xABBA+int64(n))

			scalar := make([]float32, n)

			for index, value := range src {
				scalar[index] = float32(math.Abs(float64(value)))
			}

			neon := make([]float32, n)
			AbsFloat32NEONAsm(&neon[0], &src[0], n)

			assertBitwiseEqual(t, "abs", scalar, neon)
		})
	}
}

func TestNegFloat32NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomFloat32Slice(n, 0xBADBAD+int64(n))

			scalar := make([]float32, n)

			for index, value := range src {
				scalar[index] = -value
			}

			neon := make([]float32, n)
			NegFloat32NEONAsm(&neon[0], &src[0], n)

			assertBitwiseEqual(t, "neg", scalar, neon)
		})
	}
}

func TestSqrtFloat32NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			// Use non-negative inputs so the comparison is well-defined.
			// IEEE sqrt of negative is NaN with a specific bit pattern;
			// FSQRT and math.Sqrt produce the same NaN pattern but the
			// non-negative regime is the operationally interesting one.
			src := randomNonNegativeFloat32Slice(n, 0xC0DE+int64(n))

			scalar := make([]float32, n)

			for index, value := range src {
				scalar[index] = float32(math.Sqrt(float64(value)))
			}

			neon := make([]float32, n)
			SqrtFloat32NEONAsm(&neon[0], &src[0], n)

			assertBitwiseEqual(t, "sqrt", scalar, neon)
		})
	}
}

func randomNonNegativeFloat32Slice(n int, seed int64) []float32 {
	out := randomFloat32Slice(n, seed)

	for index, value := range out {
		if value < 0 {
			out[index] = -value
		}
	}

	return out
}

func TestReluFloat32NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomFloat32Slice(n, 0xDEADBEEF+int64(n))

			scalar := make([]float32, n)

			for index, value := range src {
				scalar[index] = 0

				if value > 0 {
					scalar[index] = value
				}
			}

			neon := make([]float32, n)
			ReluFloat32NEONAsm(&neon[0], &src[0], n)

			assertBitwiseEqual(t, "relu", scalar, neon)
		})
	}
}

func BenchmarkReluFloat32NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := randomFloat32Slice(n, 1)
			dst := make([]float32, n)

			b.SetBytes(int64(n * 4 * 2))
			b.ResetTimer()

			for b.Loop() {
				ReluFloat32NEONAsm(&dst[0], &src[0], n)
			}
		})
	}
}

func BenchmarkReluFloat32Scalar(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := randomFloat32Slice(n, 1)
			dst := make([]float32, n)

			b.SetBytes(int64(n * 4 * 2))
			b.ResetTimer()

			for b.Loop() {
				for index, value := range src {
					dst[index] = 0

					if value > 0 {
						dst[index] = value
					}
				}
			}
		})
	}
}

func BenchmarkAbsFloat32NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := randomFloat32Slice(n, 1)
			dst := make([]float32, n)

			b.SetBytes(int64(n * 4 * 2))
			b.ResetTimer()

			for b.Loop() {
				AbsFloat32NEONAsm(&dst[0], &src[0], n)
			}
		})
	}
}

func BenchmarkSqrtFloat32NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := randomNonNegativeFloat32Slice(n, 1)
			dst := make([]float32, n)

			b.SetBytes(int64(n * 4 * 2))
			b.ResetTimer()

			for b.Loop() {
				SqrtFloat32NEONAsm(&dst[0], &src[0], n)
			}
		})
	}
}

func BenchmarkSqrtFloat32Scalar(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := randomNonNegativeFloat32Slice(n, 1)
			dst := make([]float32, n)

			b.SetBytes(int64(n * 4 * 2))
			b.ResetTimer()

			for b.Loop() {
				for index, value := range src {
					dst[index] = float32(math.Sqrt(float64(value)))
				}
			}
		})
	}
}
