package activation

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/tensor"
)

func TestPackedGatedShape(t *testing.T) {
	cases := []struct {
		label     string
		dims      []int
		wantBatch int
		wantHalf  int
		wantOK    bool
	}{
		{"vector_even", []int{8}, 1, 4, true},
		{"matrix", []int{3, 64}, 3, 32, true},
		{"rank3", []int{2, 3, 1024}, 6, 512, true},
		{"odd_last_dim", []int{3, 7}, 0, 0, false},
	}

	convey.Convey("Given PackedGatedShape cases", t, func() {
		for _, testCase := range cases {
			convey.Convey(testCase.label, func() {
				shape, err := tensor.NewShape(testCase.dims)

				convey.So(err, convey.ShouldBeNil)

				batch, halfCount, ok := PackedGatedShape(shape)
				convey.So(ok, convey.ShouldEqual, testCase.wantOK)

				if !testCase.wantOK {
					return
				}

				convey.So(batch, convey.ShouldEqual, testCase.wantBatch)
				convey.So(halfCount, convey.ShouldEqual, testCase.wantHalf)
			})
		}
	})
}
