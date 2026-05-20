package model_editing

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func TestWeightGraftAddFloat32NativeParityLengths(t *testing.T) {
	convey.Convey("Given WeightGraftAddFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match WeightGraftAddFloat32Scalar for N=%d", length), func() {
				weights, injection := randomGraftVectors(length, 0x2B00+int64(length))
				got := append([]float32(nil), weights...)
				want := append([]float32(nil), weights...)

				WeightGraftAddFloat32Native(got, injection)
				WeightGraftAddFloat32Scalar(want, injection)

				assertFloat32SliceEqual(t, got, want)
			})
		}
	})
}
