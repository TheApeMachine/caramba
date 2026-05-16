//go:build darwin && cgo

package metal

import . "github.com/smartystreets/goconvey/convey"

func assertAlmostEqualSlice(actual []float64, expected []float64, tolerance float64) {
	So(len(actual), ShouldEqual, len(expected))

	for index, value := range actual {
		So(value, ShouldAlmostEqual, expected[index], tolerance)
	}
}
