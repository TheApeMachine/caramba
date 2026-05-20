package interpretability

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func TestActivationSteerFloat32NativeParityLengths(t *testing.T) {
	convey.Convey("Given ActivationSteerFloat32Native", t, func() {
		const coefficient = float32(0.25)

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match ActivationSteerFloat32Scalar for N=%d", length), func() {
				base, direction := randomSteerVectors(length, 0x2A00+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				ActivationSteerFloat32Native(got, base, direction, coefficient)
				ActivationSteerFloat32Scalar(want, base, direction, coefficient)

				assertFloat32SliceEqual(t, got, want)
			})
		}
	})
}
