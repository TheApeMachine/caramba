//go:build arm64

package kernels

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

/*
Parity + benchmark suite for the NEON elementwise float32 kernels. The
scalar reference is the loop body used on architectures without a SIMD
path. Tolerance is exact bitwise — the underlying NEON instructions
(VFADD/VFSUB/VFMUL) operate on IEEE 754 binary32 with the same rounding
mode as Go's scalar +, -, * so the result must be bit-identical.
*/

var elementwiseParityNs = []int{1, 7, 64, 1024, 8192}

func TestAddFloat32NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomFloat32Slice(n, 0xA11CE+int64(n))
			right := randomFloat32Slice(n, 0xB0B+int64(n))

			scalar := make([]float32, n)

			for index := range scalar {
				scalar[index] = left[index] + right[index]
			}

			neon := make([]float32, n)
			addFloat32NEONAsm(&neon[0], &left[0], &right[0], n)

			assertBitwiseEqual(t, "add", scalar, neon)
		})
	}
}

func TestSubFloat32NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomFloat32Slice(n, 0xC0FFEE+int64(n))
			right := randomFloat32Slice(n, 0xDEAF+int64(n))

			scalar := make([]float32, n)

			for index := range scalar {
				scalar[index] = left[index] - right[index]
			}

			neon := make([]float32, n)
			subFloat32NEONAsm(&neon[0], &left[0], &right[0], n)

			assertBitwiseEqual(t, "sub", scalar, neon)
		})
	}
}

func TestDivFloat32NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomFloat32Slice(n, 0x1234+int64(n))
			// Right operand uses a separate seed and is bumped away from
			// zero so the test exercises normalized FDIV without
			// inf-from-zero divergences that would only verify NaN
			// propagation rather than the kernel itself.
			right := randomNonZeroFloat32Slice(n, 0x5678+int64(n))

			scalar := make([]float32, n)

			for index := range scalar {
				scalar[index] = left[index] / right[index]
			}

			neon := make([]float32, n)
			divFloat32NEONAsm(&neon[0], &left[0], &right[0], n)

			assertBitwiseEqual(t, "div", scalar, neon)
		})
	}
}

func randomNonZeroFloat32Slice(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, n)

	for index := range out {
		value := float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))

		if math.Abs(float64(value)) < 1e-10 {
			value = 1
		}

		out[index] = value
	}

	return out
}

func TestMulFloat32NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomFloat32Slice(n, 0xFEED+int64(n))
			right := randomFloat32Slice(n, 0xBEEF+int64(n))

			scalar := make([]float32, n)

			for index := range scalar {
				scalar[index] = left[index] * right[index]
			}

			neon := make([]float32, n)
			mulFloat32NEONAsm(&neon[0], &left[0], &right[0], n)

			assertBitwiseEqual(t, "mul", scalar, neon)
		})
	}
}

func randomFloat32Slice(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, n)

	for index := range out {
		// Span normals across a few decades and include negatives.
		out[index] = float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))
	}

	return out
}

func assertBitwiseEqual(t *testing.T, op string, scalar, neon []float32) {
	t.Helper()

	for index := range scalar {
		if math.Float32bits(scalar[index]) == math.Float32bits(neon[index]) {
			continue
		}

		t.Fatalf("%s: lane %d scalar=%g (0x%08x) neon=%g (0x%08x)",
			op,
			index,
			scalar[index], math.Float32bits(scalar[index]),
			neon[index], math.Float32bits(neon[index]),
		)
	}
}

func BenchmarkAddFloat32NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			left := randomFloat32Slice(n, 1)
			right := randomFloat32Slice(n, 2)
			out := make([]float32, n)

			b.SetBytes(int64(n * 4 * 3))
			b.ResetTimer()

			for b.Loop() {
				addFloat32NEONAsm(&out[0], &left[0], &right[0], n)
			}
		})
	}
}

func BenchmarkAddFloat32Scalar(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			left := randomFloat32Slice(n, 1)
			right := randomFloat32Slice(n, 2)
			out := make([]float32, n)

			b.SetBytes(int64(n * 4 * 3))
			b.ResetTimer()

			for b.Loop() {
				for index := range out {
					out[index] = left[index] + right[index]
				}
			}
		})
	}
}

func BenchmarkMulFloat32NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			left := randomFloat32Slice(n, 1)
			right := randomFloat32Slice(n, 2)
			out := make([]float32, n)

			b.SetBytes(int64(n * 4 * 3))
			b.ResetTimer()

			for b.Loop() {
				mulFloat32NEONAsm(&out[0], &left[0], &right[0], n)
			}
		})
	}
}
