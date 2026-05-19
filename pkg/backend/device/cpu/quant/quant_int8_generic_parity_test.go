package quant

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func referenceQuantInt8(dst []int8, src []float32, scale float32, zeroPoint int8) {
	for index, value := range src {
		scaled := math.Round(float64(value/scale)) + float64(zeroPoint)

		switch {
		case scaled > float64(math.MaxInt8):
			dst[index] = math.MaxInt8
		case scaled < float64(math.MinInt8):
			dst[index] = math.MinInt8
		default:
			dst[index] = int8(scaled)
		}
	}
}

func TestQuantInt8GenericParity(t *testing.T) {
	convey.Convey("Given quantInt8Generic", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match the reference quant for N=%d", length), func() {
				rng := rand.New(rand.NewSource(0x55aa + int64(length)))
				source := make([]float32, length)

				for index := range source {
					source[index] = float32(rng.NormFloat64()) * 10
				}

				const scale = float32(0.125)
				const zeroPoint = int8(7)

				want := make([]int8, length)
				got := make([]int8, length)

				referenceQuantInt8(want, source, scale, zeroPoint)
				quantInt8Generic(got, source, scale, zeroPoint)

				for index := range want {
					if want[index] != got[index] {
						t.Fatalf(
							"N=%d lane %d want=%d got=%d src=%g",
							length, index, want[index], got[index], source[index],
						)
					}
				}
			})
		}
	})
}

func BenchmarkQuantInt8Generic(b *testing.B) {
	const length = 8192

	source := make([]float32, length)
	rng := rand.New(rand.NewSource(1))

	for index := range source {
		source[index] = float32(rng.NormFloat64()) * 10
	}

	destination := make([]int8, length)

	b.SetBytes(int64(length * 5))
	b.ResetTimer()

	for b.Loop() {
		quantInt8Generic(destination, source, 0.125, 7)
	}
}
