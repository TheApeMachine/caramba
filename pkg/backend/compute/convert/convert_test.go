package convert

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var parityNs = []int{1, 7, 64, 1024, 8192}

/*
ulpFloat32 returns the magnitude of one unit in the last place of the
given float32. For zero, returns the smallest subnormal magnitude.
*/
func ulpFloat32(value float32) float32 {
	if value == 0 {
		return math.SmallestNonzeroFloat32
	}

	next := math.Nextafter(float64(value), math.Inf(1))
	return float32(math.Abs(next - float64(value)))
}

/*
ulpFloat64 returns the magnitude of one unit in the last place of the
given float64.
*/
func ulpFloat64(value float64) float64 {
	if value == 0 {
		return math.SmallestNonzeroFloat64
	}

	return math.Abs(math.Nextafter(value, math.Inf(1)) - value)
}

/*
bf16ULP returns the ULP magnitude in the BF16 grid for a value: the
gap between consecutive bf16 values straddling the target. BF16 has
8-bit exponent + 7-bit mantissa, so one ULP at magnitude v ≈ |v| ×
2^-7 (≈ |v| / 128) within the normal range.
*/
func bf16ULP(value float32) float32 {
	if value == 0 {
		// Smallest non-zero bf16 subnormal ≈ 2^-133 (approx).
		return math.SmallestNonzeroFloat32
	}

	abs := float32(math.Abs(float64(value)))
	return abs / 128
}

func TestBFloat16ToFloat32_RoundTrip(t *testing.T) {
	for _, n := range parityNs {
		n := n

		t.Run("N="+itoaSimple(n), func(t *testing.T) {
			convey.Convey("Round-trip should land within 1 bf16 ULP per element", t, func() {
				original := make([]float32, n)

				for index := range original {
					original[index] = float32(index%19) * 0.5
				}

				bf16s := make([]dtype.BF16, len(original))
				err := Float32ToBFloat16(bf16s, original)
				convey.So(err, convey.ShouldBeNil)

				roundTrip := make([]float32, len(original))
				err = BFloat16ToFloat32(roundTrip, bf16s)
				convey.So(err, convey.ShouldBeNil)

				for index, expected := range original {
					tolerance := bf16ULP(expected)
					convey.So(roundTrip[index], convey.ShouldAlmostEqual, expected, tolerance)
				}
			})
		})
	}
}

func TestFloat16ToFloat32_RoundTrip(t *testing.T) {
	for _, n := range parityNs {
		n := n

		t.Run("N="+itoaSimple(n), func(t *testing.T) {
			convey.Convey("Round-trip should land within 1 f16 ULP per element", t, func() {
				original := make([]float32, n)

				for index := range original {
					// Stay within the f16 representable range.
					original[index] = float32((index%13)-6) * 0.25
				}

				f16s := make([]dtype.F16, len(original))
				err := Float32ToFloat16(f16s, original)
				convey.So(err, convey.ShouldBeNil)

				roundTrip := make([]float32, len(original))
				err = Float16ToFloat32(roundTrip, f16s)
				convey.So(err, convey.ShouldBeNil)

				for index, expected := range original {
					// 1 f16 ULP at magnitude v ≈ |v| × 2^-10.
					tolerance := float32(math.Abs(float64(expected))) / 1024

					if tolerance == 0 {
						tolerance = math.SmallestNonzeroFloat32
					}

					convey.So(roundTrip[index], convey.ShouldAlmostEqual, expected, tolerance)
				}
			})
		})
	}
}

func TestFloat32ToFloat64_AndBack(t *testing.T) {
	for _, n := range parityNs {
		n := n

		t.Run("N="+itoaSimple(n), func(t *testing.T) {
			convey.Convey("Float32→Float64→Float32 should be exact", t, func() {
				original := make([]float32, n)

				for index := range original {
					original[index] = float32(index%17) * 0.125
				}

				float64s := make([]float64, len(original))
				err := Float32ToFloat64(float64s, original)
				convey.So(err, convey.ShouldBeNil)

				roundTrip := make([]float32, len(original))
				err = Float64ToFloat32(roundTrip, float64s)
				convey.So(err, convey.ShouldBeNil)

				convey.So(roundTrip, convey.ShouldResemble, original)
			})
		})
	}
}

func TestFloat8E4M3_RoundTrip(t *testing.T) {
	for _, n := range parityNs {
		n := n

		t.Run("N="+itoaSimple(n), func(t *testing.T) {
			convey.Convey("Round-trip should land within fp8 ULP per element", t, func() {
				original := make([]float32, n)

				for index := range original {
					original[index] = float32(index%9-4) * 0.5
				}

				fp8s := make([]dtype.F8E4M3, len(original))
				err := Float32ToFloat8E4M3(fp8s, original)
				convey.So(err, convey.ShouldBeNil)

				roundTrip := make([]float32, len(original))
				err = Float8E4M3ToFloat32(roundTrip, fp8s)
				convey.So(err, convey.ShouldBeNil)

				for index, expected := range original {
					// fp8 e4m3 has 3 mantissa bits ≈ |v| × 2^-3.
					tolerance := float32(math.Abs(float64(expected))) / 8

					if tolerance < 0.0625 {
						tolerance = 0.0625
					}

					convey.So(roundTrip[index], convey.ShouldAlmostEqual, expected, tolerance)
				}
			})
		})
	}
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

// itoaSimple is a minimal int→string helper; strconv import would
// pull in additional dependencies in this test file.
func itoaSimple(n int) string {
	if n == 0 {
		return "0"
	}

	negative := false

	if n < 0 {
		negative = true
		n = -n
	}

	var digits [20]byte
	index := len(digits)

	for n > 0 {
		index--
		digits[index] = byte('0' + n%10)
		n /= 10
	}

	if negative {
		index--
		digits[index] = '-'
	}

	return string(digits[index:])
}

func BenchmarkBFloat16ToFloat32_1024(b *testing.B) {
	src := make([]dtype.BF16, 1024)
	dst := make([]float32, 1024)

	for index := range src {
		src[index] = dtype.NewBfloat16FromFloat32(float32(index) * 0.5)
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(len(src) * 2))

	for b.Loop() {
		_ = BFloat16ToFloat32(dst, src)
	}
}

func BenchmarkFloat32ToBFloat16_1024(b *testing.B) {
	src := make([]float32, 1024)
	dst := make([]dtype.BF16, 1024)

	for index := range src {
		src[index] = float32(index) * 0.5
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(len(src) * 4))

	for b.Loop() {
		_ = Float32ToBFloat16(dst, src)
	}
}

func BenchmarkFloat32ToFloat64_1024(b *testing.B) {
	src := make([]float32, 1024)
	dst := make([]float64, 1024)

	for index := range src {
		src[index] = float32(index) * 0.5
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(len(src) * 4))

	for b.Loop() {
		_ = Float32ToFloat64(dst, src)
	}
}

func BenchmarkFloat8E4M3ToFloat32_1024(b *testing.B) {
	src := make([]dtype.F8E4M3, 1024)
	dst := make([]float32, 1024)

	for index := range src {
		src[index] = dtype.NewF8E4M3FromFloat32(float32(index%9-4) * 0.5)
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(len(src)))

	for b.Loop() {
		_ = Float8E4M3ToFloat32(dst, src)
	}
}
