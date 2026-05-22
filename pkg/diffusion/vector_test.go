package diffusion

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestCompareVectors(testingObject *testing.T) {
	convey.Convey("Given two different vectors", testingObject, func() {
		left := []float32{1, 0, 0}
		right := []float32{0, 1, 0}

		convey.Convey("It should report non-zero L2 and max-abs", func() {
			l2, maxAbs, err := CompareVectors(left, right)

			convey.So(err, convey.ShouldBeNil)
			convey.So(l2, convey.ShouldAlmostEqual, 1.4142135, 1e-5)
			convey.So(maxAbs, convey.ShouldEqual, float32(1))
		})
	})

	convey.Convey("Given identical vectors", testingObject, func() {
		values := []float32{2, 3, 5}

		convey.Convey("It should report zero difference", func() {
			l2, maxAbs, err := CompareVectors(values, values)

			convey.So(err, convey.ShouldBeNil)
			convey.So(l2, convey.ShouldEqual, 0)
			convey.So(maxAbs, convey.ShouldEqual, float32(0))
		})
	})
}
