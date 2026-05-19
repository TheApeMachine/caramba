package parity

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestFloat32ULPDistance(t *testing.T) {
	Convey("Given adjacent float32 values", t, func() {
		Convey("It should report one ULP apart", func() {
			So(Float32ULPDistance(1.0, math.Nextafter32(1.0, 2)), ShouldEqual, 1)
		})
	})
}

func TestAssertFloat32SlicesWithinULP(t *testing.T) {
	AssertFloat32SlicesWithinULP(t, []float32{1, 2}, []float32{1, 2}, 1)
}

func TestFloat32LanesMatchNearZero(t *testing.T) {
	AssertFloat32SlicesWithinULP(t, []float32{0}, []float32{-1e-12}, 2)
}

func BenchmarkFloat32ULPDistance(b *testing.B) {
	left := float32(1.0)
	right := float32(1.1)

	for b.Loop() {
		_ = Float32ULPDistance(left, right)
	}
}
