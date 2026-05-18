package metal

import (
	"context"
	"errors"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestNewBackend_Stub(t *testing.T) {
	convey.Convey("On non-darwin or no-cgo builds", t, func() {
		_, err := NewBackend()

		convey.Convey("It should return ErrNeedsPlatformSetup", func() {
			convey.So(errors.Is(err, tensor.ErrNeedsPlatformSetup), convey.ShouldBeTrue)
		})
	})
}

func TestBackend_Location(t *testing.T) {
	convey.Convey("Location should report Metal regardless of stub status", t, func() {
		backend := &Backend{}
		convey.So(backend.Location(), convey.ShouldEqual, tensor.Metal)
	})
}

func TestBackend_SupportedDTypes(t *testing.T) {
	convey.Convey("SupportedDTypes should return Metal-native dtypes", t, func() {
		backend := &Backend{}
		dtypes := backend.SupportedDTypes()

		convey.So(dtypes, convey.ShouldContain, dtype.Float32)
		convey.So(dtypes, convey.ShouldContain, dtype.BFloat16)
		convey.So(dtypes, convey.ShouldContain, dtype.Float16)
	})
}

func TestBackend_SupportedLayouts(t *testing.T) {
	convey.Convey("SupportedLayouts should include LayoutDense", t, func() {
		backend := &Backend{}
		layouts := backend.SupportedLayouts()
		convey.So(layouts, convey.ShouldContain, tensor.LayoutDense)
	})
}

func TestBackend_Capabilities(t *testing.T) {
	convey.Convey("Capabilities should report Apple-recommended alignment", t, func() {
		backend := &Backend{}
		caps := backend.Capabilities()
		convey.So(caps.NativeAlignment, convey.ShouldEqual, 256)
	})
}

func TestBackend_UploadVariants_Stub(t *testing.T) {
	convey.Convey("Upload paths should error cleanly when no bridge is present", t, func() {
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
	convey.Convey("Download should error when no bridge is present", t, func() {
		backend := &Backend{}
		_, _, err := backend.Download(nil)
		convey.So(errors.Is(err, tensor.ErrNeedsPlatformSetup), convey.ShouldBeTrue)
	})
}

func TestBackend_Close(t *testing.T) {
	convey.Convey("Close should be idempotent and never error on a stub", t, func() {
		backend := &Backend{}
		convey.So(backend.Close(), convey.ShouldBeNil)
		convey.So(backend.Close(), convey.ShouldBeNil)
	})
}

func TestSyncBlocking_NilTensor(t *testing.T) {
	convey.Convey("SyncBlocking on a nil tensor panics, but the surface compiles", t, func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected panic on nil tensor")
			}
		}()

		_ = SyncBlocking(context.Background(), nil)
	})
}

func BenchmarkNewBackend_Stub(b *testing.B) {
	for b.Loop() {
		_, _ = NewBackend()
	}
}

func BenchmarkBackend_Location(b *testing.B) {
	backend := &Backend{}

	for b.Loop() {
		_ = backend.Location()
	}
}

func BenchmarkBackend_Close(b *testing.B) {
	for b.Loop() {
		backend := &Backend{}
		_ = backend.Close()
	}
}
