package dequant

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func referenceDequantInt8(dst []float32, src []int8, scale float32, zeroPoint int8) {
	for index, value := range src {
		dst[index] = float32(int32(value)-int32(zeroPoint)) * scale
	}
}

func TestDequantInt8GenericParity(t *testing.T) {
	convey.Convey("Given dequantInt8Generic", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match the reference dequant for N=%d", length), func() {
				rng := rand.New(rand.NewSource(0x401d + int64(length)))
				source := make([]int8, length)

				for index := range source {
					source[index] = int8(rng.Intn(256) - 128)
				}

				const scale = float32(0.0875)
				const zeroPoint = int8(-13)

				want := make([]float32, length)
				got := make([]float32, length)

				referenceDequantInt8(want, source, scale, zeroPoint)
				dequantInt8Generic(got, source, scale, zeroPoint)

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
	})
}

func BenchmarkDequantInt8Generic(b *testing.B) {
	const length = 8192

	source := make([]int8, length)
	rng := rand.New(rand.NewSource(1))

	for index := range source {
		source[index] = int8(rng.Intn(256) - 128)
	}

	destination := make([]float32, length)

	b.SetBytes(int64(length * 5))
	b.ResetTimer()

	for b.Loop() {
		dequantInt8Generic(destination, source, 0.0875, -13)
	}
}
