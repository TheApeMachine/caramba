package tokenizer

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func TestPackInt32NativeParityLengths(t *testing.T) {
	convey.Convey("Given PackInt32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PackInt32Scalar for N=%d", length), func() {
				source := randomInt32Vector(length, 0x2800+int64(length))
				got := make([]int32, length)
				want := make([]int32, length)

				PackInt32Native(got, source)
				PackInt32Scalar(want, source)

				assertInt32SliceEqual(t, got, want)
			})
		}
	})
}
