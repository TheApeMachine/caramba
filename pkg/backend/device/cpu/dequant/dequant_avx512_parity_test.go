//go:build amd64

package dequant

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512DequantAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestDequantInt8AVX512Parity(t *testing.T) {
	if !avx512DequantAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given DequantInt8AVX512Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match dequantInt8Generic for N=%d", length), func() {
				rng := rand.New(rand.NewSource(0x401d + int64(length)))
				source := make([]int8, length)

				for index := range source {
					source[index] = int8(rng.Intn(256) - 128)
				}

				const scale = float32(0.0875)
				const zeroPoint = int8(-13)

				want := make([]float32, length)
				got := make([]float32, length)

				dequantInt8Generic(want, source, scale, zeroPoint)
				dequantInt8AVX512(got, source, scale, zeroPoint)

				for index := range want {
					if want[index] != got[index] {
						t.Fatalf(
							"N=%d lane %d want=%g got=%g q=%d",
							length, index, want[index], got[index], source[index],
						)
					}
				}
			})
		}

		convey.Convey("It should match dequantInt8Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				rng := rand.New(rand.NewSource(0x401e + int64(length)))
				source := make([]int8, length)

				for index := range source {
					source[index] = int8(rng.Intn(256) - 128)
				}

				const scale = float32(0.0875)
				const zeroPoint = int8(-13)

				want := make([]float32, length)
				got := make([]float32, length)

				dequantInt8Generic(want, source, scale, zeroPoint)
				DequantInt8AVX512Asm(
					&got[0], &source[0], length,
					scale, int16(zeroPoint),
				)

				for index := range want {
					if want[index] != got[index] {
						t.Fatalf(
							"N=%d lane %d want=%g got=%g q=%d",
							length, index, want[index], got[index], source[index],
						)
					}
				}
			}
		})
	})
}

func TestDequantInt4AVX512Parity(t *testing.T) {
	if !avx512DequantAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given DequantInt4AVX512Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match dequantInt4Generic for N=%d", length), func() {
				bytes := int4BytesFromLength(length, 0x401e+int64(length))
				pairs := int4VectorFromBytes(bytes, length)

				const scale = float32(0.0625)
				const zeroPoint = int8(3)

				want := make([]float32, length)
				got := make([]float32, length)

				dequantInt4Generic(want, pairs, scale, zeroPoint)
				dequantInt4AVX512(got, pairs.Bytes(), length, scale, zeroPoint)

				for index := range want {
					if want[index] != got[index] {
						t.Fatalf(
							"N=%d lane %d want=%g got=%g nibble=%d",
							length, index, want[index], got[index], pairs.Get(index),
						)
					}
				}
			})
		}

		convey.Convey("It should match dequantInt4Generic via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				bytes := int4BytesFromLength(length, 0x401f+int64(length))
				pairs := int4VectorFromBytes(bytes, length)

				const scale = float32(0.0625)
				const zeroPoint = int8(3)

				want := make([]float32, length)
				got := make([]float32, length)

				dequantInt4Generic(want, pairs, scale, zeroPoint)
				DequantInt4AVX512Asm(
					&got[0], &bytes[0], length,
					scale, zeroPoint,
				)

				for index := range want {
					if want[index] != got[index] {
						t.Fatalf(
							"N=%d lane %d want=%g got=%g nibble=%d",
							length, index, want[index], got[index], pairs.Get(index),
						)
					}
				}
			}
		})
	})
}
