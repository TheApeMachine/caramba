package peel

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestSimdLaneCount(t *testing.T) {
	Convey("Given ISA names", t, func() {
		Convey("It should return vector lane counts", func() {
			So(SimdLaneCount("avx512"), ShouldEqual, 16)
			So(SimdLaneCount("avx2"), ShouldEqual, 8)
			So(SimdLaneCount("sse2"), ShouldEqual, 4)
		})
	})
}

func TestWrapF32Unary(t *testing.T) {
	Convey("Given peel wrappers", t, func() {
		var simdCount, genericCount int

		simdKernel := func(dst, src *float32, count int) {
			simdCount = count
		}

		genericKernel := func(dst, src *float32, count int) {
			genericCount = count
		}

		wrapped := WrapF32Unary(simdKernel, genericKernel, "avx2")

		Convey("It should split 10 elements into 8 SIMD and 2 generic", func() {
			var dst, src float32
		wrapped(&dst, &src, 10)

			So(simdCount, ShouldEqual, 8)
			So(genericCount, ShouldEqual, 2)
		})
	})
}

func BenchmarkWrapF32Unary(b *testing.B) {
	buffer := make([]float32, 8192)
	wrapped := WrapF32Unary(
		func(dst, src *float32, count int) {},
		func(dst, src *float32, count int) {},
		"avx2",
	)

	for b.Loop() {
		wrapped(&buffer[0], &buffer[0], len(buffer))
	}
}
