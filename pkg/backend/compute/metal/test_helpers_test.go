//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"

	. "github.com/smartystreets/goconvey/convey"
)

func assertAlmostEqualSlice(actual []float64, expected []float64, tolerance float64) {
	So(len(actual), ShouldEqual, len(expected))

	for index, value := range actual {
		So(value, ShouldAlmostEqual, expected[index], tolerance)
	}
}

func assertMetalMaxDiff(actual []float64, expected []float64, tolerance float64) {
	So(actual, ShouldHaveLength, len(expected))

	if len(expected) == 0 {
		return
	}

	maxDiff := 0.0
	maxIndex := 0

	for index, value := range actual {
		diff := math.Abs(value - expected[index])

		if diff <= maxDiff {
			continue
		}

		maxDiff = diff
		maxIndex = index
	}

	SoMsg(
		fmt.Sprintf(
			"max_diff=%g index=%d actual=%g expected=%g tolerance=%g",
			maxDiff,
			maxIndex,
			actual[maxIndex],
			expected[maxIndex],
			tolerance,
		),
		maxDiff <= tolerance,
		ShouldBeTrue,
	)
}
