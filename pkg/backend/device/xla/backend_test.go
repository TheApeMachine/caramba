package xla

import (
	"errors"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestNewBackend_Stub(t *testing.T) {
	convey.Convey("On builds without the 'xla' tag", t, func() {
		_, err := NewBackend()

		convey.Convey("It should return ErrNeedsPlatformSetup", func() {
			convey.So(errors.Is(err, tensor.ErrNeedsPlatformSetup), convey.ShouldBeTrue)
		})
	})
}
