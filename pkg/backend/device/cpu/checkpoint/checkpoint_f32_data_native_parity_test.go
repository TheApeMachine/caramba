package checkpoint

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func TestEncodeFloat32DataNativeParityLengths(t *testing.T) {
	convey.Convey("Given EncodeFloat32DataNative", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match encodeFloat32DataScalar for N=%d", length), func() {
				source := randomFloat32Vector(length, 0x2900+int64(length))
				got := make([]uint8, length*4)
				want := make([]uint8, length*4)

				EncodeFloat32DataNative(got, source)
				encodeFloat32DataScalar(want, source)

				assertUint8PayloadEqual(t, got, want)
			})
		}
	})
}

func TestDecodeFloat32DataNativeParityLengths(t *testing.T) {
	convey.Convey("Given DecodeFloat32DataNative", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match decodeFloat32DataScalar for N=%d", length), func() {
				source := randomFloat32Vector(length, 0x2910+int64(length))
				payload := make([]uint8, length*4)
				encodeFloat32DataScalar(payload, source)

				got := make([]float32, length)
				want := make([]float32, length)

				DecodeFloat32DataNative(got, payload)
				decodeFloat32DataScalar(want, payload)

				assertFloat32SliceEqual(t, got, want)
			})
		}
	})
}
