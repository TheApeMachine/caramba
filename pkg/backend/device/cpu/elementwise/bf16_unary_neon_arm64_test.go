//go:build arm64

package elementwise

import (
	"fmt"
	"math"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

func TestAbsBFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomBF16Slice(n, 0x9000+int64(n))

			scalar := make([]dtype.BF16, n)

			for index := range scalar {
				value := (&src[index]).Float32()
				scalar[index] = dtype.NewBfloat16FromFloat32(float32(math.Abs(float64(value))))
			}

			neon := make([]dtype.BF16, n)
			AbsBFloat16Native(neon, src)

			assertBF16Equal(t, "abs", scalar, neon)
		})
	}
}

func TestNegBFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomBF16Slice(n, 0xA000+int64(n))

			scalar := make([]dtype.BF16, n)

			for index := range scalar {
				value := (&src[index]).Float32()
				scalar[index] = dtype.NewBfloat16FromFloat32(-value)
			}

			neon := make([]dtype.BF16, n)
			NegBFloat16Native(neon, src)

			assertBF16Equal(t, "neg", scalar, neon)
		})
	}
}

func TestSqrtBFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomBF16Slice(n, 0xB000+int64(n))

			// FSQRT of negative is NaN with a hardware-specific pattern.
			// To keep parity meaningful we force non-negative inputs.
			for index := range src {
				src[index] = dtype.BF16(uint16(src[index]) & 0x7FFF)
			}

			scalar := make([]dtype.BF16, n)

			for index := range scalar {
				value := (&src[index]).Float32()
				scalar[index] = dtype.NewBfloat16FromFloat32(float32(math.Sqrt(float64(value))))
			}

			neon := make([]dtype.BF16, n)
			SqrtBFloat16Native(neon, src)

			assertBF16Equal(t, "sqrt", scalar, neon)
		})
	}
}

func TestReluBFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomBF16Slice(n, 0xC000+int64(n))

			scalar := make([]dtype.BF16, n)

			for index := range scalar {
				value := (&src[index]).Float32()
				scalar[index] = dtype.NewBfloat16FromFloat32(0)

				if value > 0 {
					scalar[index] = dtype.NewBfloat16FromFloat32(value)
				}
			}

			neon := make([]dtype.BF16, n)
			ReluBFloat16Native(neon, src)

			assertBF16Equal(t, "relu", scalar, neon)
		})
	}
}
