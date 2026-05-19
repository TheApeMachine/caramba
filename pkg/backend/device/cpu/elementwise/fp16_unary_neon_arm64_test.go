//go:build arm64

package elementwise

import (
	"fmt"
	"math"
	"testing"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestAbsFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomF16Slice(n, 0x1010+int64(n))

			scalar := make([]dtype.F16, n)
			for index := range scalar {
				scalar[index] = dtype.Fromfloat32(float32(math.Abs(float64(src[index].Float32()))))
			}

			neon := make([]dtype.F16, n)
			AbsFloat16Native(neon, src)

			assertF16Equal(t, "abs", scalar, neon)
		})
	}
}

func TestNegFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomF16Slice(n, 0x2020+int64(n))

			scalar := make([]dtype.F16, n)
			for index := range scalar {
				scalar[index] = dtype.Fromfloat32(-src[index].Float32())
			}

			neon := make([]dtype.F16, n)
			NegFloat16Native(neon, src)

			assertF16Equal(t, "neg", scalar, neon)
		})
	}
}

func TestSqrtFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomF16Slice(n, 0x3030+int64(n))
			for index := range src {
				src[index] = dtype.F16(uint16(src[index]) & 0x7FFF)
			}

			scalar := make([]dtype.F16, n)
			for index := range scalar {
				scalar[index] = dtype.Fromfloat32(float32(math.Sqrt(float64(src[index].Float32()))))
			}

			neon := make([]dtype.F16, n)
			SqrtFloat16Native(neon, src)

			assertF16Equal(t, "sqrt", scalar, neon)
		})
	}
}

func TestReluFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			src := randomF16Slice(n, 0x4040+int64(n))

			scalar := make([]dtype.F16, n)
			for index := range scalar {
				value := src[index].Float32()
				scalar[index] = dtype.Fromfloat32(0)

				if value > 0 {
					scalar[index] = dtype.Fromfloat32(value)
				}
			}

			neon := make([]dtype.F16, n)
			ReluFloat16Native(neon, src)

			assertF16Equal(t, "relu", scalar, neon)
		})
	}
}
