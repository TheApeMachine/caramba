package cuda

import (
	"errors"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestNewBackend_Stub(t *testing.T) {
	convey.Convey("On builds without the 'cuda' tag", t, func() {
		_, err := NewBackend()

		convey.Convey("It should return ErrNeedsPlatformSetup", func() {
			convey.So(errors.Is(err, tensor.ErrNeedsPlatformSetup), convey.ShouldBeTrue)
		})
	})
}

func TestBackend_Location(t *testing.T) {
	convey.Convey("Location should report CUDA", t, func() {
		backend := &Backend{}
		convey.So(backend.Location(), convey.ShouldEqual, tensor.CUDA)
	})
}

func TestBackend_SupportedDTypes_Stub(t *testing.T) {
	convey.Convey("Without a bridge, SupportedDTypes returns nil", t, func() {
		backend := &Backend{}
		convey.So(backend.SupportedDTypes(), convey.ShouldBeNil)
	})
}

func TestBackend_SupportedLayouts(t *testing.T) {
	convey.Convey("SupportedLayouts should include LayoutDense", t, func() {
		backend := &Backend{}
		convey.So(backend.SupportedLayouts(), convey.ShouldContain, tensor.LayoutDense)
	})
}

func TestBackend_Capabilities(t *testing.T) {
	convey.Convey("Capabilities should report 128-byte alignment", t, func() {
		backend := &Backend{}
		caps := backend.Capabilities()
		convey.So(caps.NativeAlignment, convey.ShouldEqual, 128)
	})
}

func TestBackend_UploadVariants_Stub(t *testing.T) {
	convey.Convey("Upload paths should return ErrNeedsPlatformSetup on the stub", t, func() {
		backend := &Backend{}
		shape, _ := tensor.NewShape([]int{4})

		_, err := backend.Upload(shape, dtype.Float32, make([]byte, 16))
		convey.So(errors.Is(err, tensor.ErrNeedsPlatformSetup), convey.ShouldBeTrue)

		_, err = backend.UploadAsync(shape, dtype.Float32, make([]byte, 16))
		convey.So(errors.Is(err, tensor.ErrNeedsPlatformSetup), convey.ShouldBeTrue)

		_, err = backend.UploadSparse(shape, dtype.Float32, tensor.LayoutSparseCSR, nil, nil)
		convey.So(errors.Is(err, tensor.ErrLayoutUnsupported), convey.ShouldBeTrue)
	})
}

func TestBackend_Download_Stub(t *testing.T) {
	convey.Convey("Download should error on the stub", t, func() {
		backend := &Backend{}
		_, _, err := backend.Download(nil)
		convey.So(errors.Is(err, tensor.ErrNeedsPlatformSetup), convey.ShouldBeTrue)
	})
}

func TestBackend_Close(t *testing.T) {
	convey.Convey("Close should be idempotent", t, func() {
		backend := &Backend{}
		convey.So(backend.Close(), convey.ShouldBeNil)
		convey.So(backend.Close(), convey.ShouldBeNil)
	})
}

func BenchmarkBackend_Location(b *testing.B) {
	backend := &Backend{}

	for b.Loop() {
		_ = backend.Location()
	}
}
