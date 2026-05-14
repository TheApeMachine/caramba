package bench

import (
	"math"
	"math/rand"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func referenceArgmax(xs []float64) int {
	if len(xs) == 0 {
		return 0
	}

	best := xs[0]
	idx := 0

	for cursor := 1; cursor < len(xs); cursor++ {
		if xs[cursor] > best {
			best = xs[cursor]
			idx = cursor
		}
	}

	return idx
}

func TestArgmax(t *testing.T) {
	Convey("Given the argmax SIMD kernel", t, func() {
		Convey("It should return 0 for an empty slice", func() {
			So(argmaxImpl(nil), ShouldEqual, 0)
			So(argmaxImpl([]float64{}), ShouldEqual, 0)
		})

		Convey("It should return 0 for a single-element slice", func() {
			So(argmaxImpl([]float64{42}), ShouldEqual, 0)
			So(argmaxImpl([]float64{-42}), ShouldEqual, 0)
		})

		Convey("It should match the scalar reference across all length classes", func() {
			rng := rand.New(rand.NewSource(7))

			for _, length := range []int{2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 1000, 1024} {
				xs := make([]float64, length)

				for index := range xs {
					xs[index] = rng.NormFloat64()
				}

				So(argmaxImpl(xs), ShouldEqual, referenceArgmax(xs))
			}
		})

		Convey("It should return the first occurrence on ties (strict >)", func() {
			xs := []float64{1, 2, 3, 3, 2, 3}

			So(argmaxImpl(xs), ShouldEqual, 2)
		})

		Convey("It should pick the global max regardless of position", func() {
			xs := make([]float64, 16)
			xs[11] = 100.0

			So(argmaxImpl(xs), ShouldEqual, 11)
		})

		Convey("It should not let NaN displace an existing best (sticky-best contract)", func() {
			xs := []float64{1.0, math.NaN(), 2.0, 0.5}

			So(argmaxImpl(xs), ShouldEqual, 2)
		})

		Convey("It should keep NaN sticky when it appears first", func() {
			xs := []float64{math.NaN(), 1.0, 2.0}

			So(argmaxImpl(xs), ShouldEqual, 0)
		})
	})
}

var benchSinkArgmax int

func BenchmarkArgmax(benchmark *testing.B) {
	rng := rand.New(rand.NewSource(42))
	xs := make([]float64, 1000)

	for index := range xs {
		xs[index] = rng.NormFloat64()
	}

	for benchmark.Loop() {
		benchSinkArgmax = argmaxImpl(xs)
	}

	_ = benchSinkArgmax
}
