package convert

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestBFloat16ToFloat32_RoundTrip(t *testing.T) {
	convey.Convey("Given values representable in BF16", t, func() {
		original := []float32{0, 1, -1, 2, 0.5, -0.5, 4.25, 256}
		bf16s := make([]dtype.BF16, len(original))
		_ = Float32ToBFloat16(bf16s, original)

		convey.Convey("BFloat16ToFloat32 should round-trip within BF16 ULP", func() {
			roundTrip := make([]float32, len(original))
			err := BFloat16ToFloat32(roundTrip, bf16s)

			convey.So(err, convey.ShouldBeNil)

			for index, expected := range original {
				convey.So(roundTrip[index], convey.ShouldAlmostEqual, expected, 1e-2)
			}
		})
	})
}

func TestFloat16ToFloat32_RoundTrip(t *testing.T) {
	convey.Convey("Given values representable in F16", t, func() {
		original := []float32{0, 1, -1, 2, 0.5, -0.5, 4.25}
		f16s := make([]dtype.F16, len(original))
		_ = Float32ToFloat16(f16s, original)

		convey.Convey("Float16ToFloat32 should round-trip exactly", func() {
			roundTrip := make([]float32, len(original))
			err := Float16ToFloat32(roundTrip, f16s)

			convey.So(err, convey.ShouldBeNil)

			for index, expected := range original {
				convey.So(roundTrip[index], convey.ShouldAlmostEqual, expected, 1e-3)
			}
		})
	})
}

func TestFloat32ToFloat64_AndBack(t *testing.T) {
	convey.Convey("Given a float32 buffer", t, func() {
		original := []float32{1.0, -2.5, 3.125}
		float64s := make([]float64, len(original))

		convey.Convey("Float32ToFloat64 should widen losslessly", func() {
			err := Float32ToFloat64(float64s, original)
			convey.So(err, convey.ShouldBeNil)

			for index, value := range original {
				convey.So(float64s[index], convey.ShouldEqual, float64(value))
			}

			roundTrip := make([]float32, len(original))
			err = Float64ToFloat32(roundTrip, float64s)
			convey.So(err, convey.ShouldBeNil)
			convey.So(roundTrip, convey.ShouldResemble, original)
		})
	})
}

func TestFloat8E4M3_RoundTrip(t *testing.T) {
	convey.Convey("Given representable values", t, func() {
		original := []float32{0, 1, -1, 0.5, 0.015625, 448}
		fp8s := make([]dtype.F8E4M3, len(original))
		_ = Float32ToFloat8E4M3(fp8s, original)

		convey.Convey("Round-trip should land within FP8 ULP", func() {
			roundTrip := make([]float32, len(original))
			err := Float8E4M3ToFloat32(roundTrip, fp8s)
			convey.So(err, convey.ShouldBeNil)

			for index, expected := range original {
				convey.So(roundTrip[index], convey.ShouldAlmostEqual, expected, 1e-2)
			}
		})
	})
}

func TestFloat32ToInt8_Saturation(t *testing.T) {
	convey.Convey("Given values outside int8 range", t, func() {
		original := []float32{1000, -1000, 50}
		ints := make([]int8, len(original))

		convey.Convey("Float32ToInt8 should saturate at the boundary", func() {
			err := Float32ToInt8(ints, original)
			convey.So(err, convey.ShouldBeNil)
			convey.So(ints, convey.ShouldResemble, []int8{math.MaxInt8, math.MinInt8, 50})
		})
	})
}

func TestInt4ToFloat32(t *testing.T) {
	convey.Convey("Given packed Int4 pairs", t, func() {
		pairs := []dtype.Int4Pair{
			dtype.NewInt4Pair(1, -2),
			dtype.NewInt4Pair(-8, 7),
		}

		convey.Convey("It should widen each nibble to float32", func() {
			float32s := make([]float32, 4)
			err := Int4ToFloat32(float32s, pairs)

			convey.So(err, convey.ShouldBeNil)
			convey.So(float32s, convey.ShouldResemble, []float32{1, -2, -8, 7})
		})
	})
}
