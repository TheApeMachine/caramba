//go:build darwin && cgo

package metal

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestNewTensorBackend(t *testing.T) {
	Convey("Given a Metal tensor backend", t, func() {
		tensorBackend := newMetalTensorBackendForTest(t)

		Convey("It should initialize resident Metal storage", func() {
			So(tensorBackend.Location(), ShouldEqual, computetensor.Metal)
		})
	})
}

func TestTensorBackend_UploadFloat64(t *testing.T) {
	Convey("Given host values", t, func() {
		tensorBackend := newMetalTensorBackendForTest(t)

		shape, err := computetensor.NewShape([]int{3})
		So(err, ShouldBeNil)

		Convey("It should upload into resident Metal storage", func() {
			uploaded, err := tensorBackend.UploadFloat64(shape, []float64{1, 2, 3})

			So(err, ShouldBeNil)
			So(uploaded.Location(), ShouldEqual, computetensor.Metal)
			So(uploaded.DType(), ShouldEqual, computetensor.Float32)
			So(uploaded.Bytes(), ShouldEqual, 12)
		})
	})
}

func TestTensorBackend_DownloadFloat64(t *testing.T) {
	Convey("Given a resident Metal tensor", t, func() {
		tensorBackend := newMetalTensorBackendForTest(t)

		shape, err := computetensor.NewShape([]int{3})
		So(err, ShouldBeNil)

		uploaded, err := tensorBackend.UploadFloat64(shape, []float64{1, 2, 3})
		So(err, ShouldBeNil)

		Convey("It should download through an explicit boundary", func() {
			values, err := tensorBackend.DownloadFloat64(uploaded)

			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2, 3})
		})
	})
}

func BenchmarkTensorBackend_UploadFloat64(benchmark *testing.B) {
	tensorBackend, err := NewTensorBackend()

	if err != nil {
		benchmark.Skip(err)
	}

	shape, err := computetensor.NewShape([]int{1024})

	if err != nil {
		benchmark.Fatal(err)
	}

	values := make([]float64, shape.Len())

	benchmark.ResetTimer()

	for iteration := 0; iteration < benchmark.N; iteration++ {
		uploaded, err := tensorBackend.UploadFloat64(shape, values)

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := uploaded.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func newMetalTensorBackendForTest(t *testing.T) *TensorBackend {
	t.Helper()

	tensorBackend, err := NewTensorBackend()

	if err != nil {
		t.Skipf("Metal tensor backend unavailable: %v", err)
	}

	return tensorBackend
}
