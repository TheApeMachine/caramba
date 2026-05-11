//go:build darwin && cgo

package metal

import (
	"sync"
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
			defer func() {
				So(uploaded.Close(), ShouldBeNil)
			}()

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
		defer func() {
			So(uploaded.Close(), ShouldBeNil)
		}()

		Convey("It should download through an explicit boundary", func() {
			values, err := tensorBackend.DownloadFloat64(uploaded)

			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2, 3})
		})
	})
}

func TestTensorBackend_CloseConcurrent(t *testing.T) {
	Convey("Given a Metal tensor backend", t, func() {
		tensorBackend := newMetalTensorBackendForTest(t)

		shape, err := computetensor.NewShape([]int{2})
		So(err, ShouldBeNil)

		Convey("Concurrent Close should be safe", func() {
			var waitGroup sync.WaitGroup

			for range 8 {
				waitGroup.Add(1)

				go func() {
					defer waitGroup.Done()
					_ = tensorBackend.Close()
				}()
			}

			waitGroup.Wait()

			_, err := tensorBackend.UploadFloat64(shape, []float64{1, 2})
			So(err, ShouldNotBeNil)
		})
	})
}

func TestTensor_Close_emptyBuffer(t *testing.T) {
	Convey("Given a zero-length Metal tensor", t, func() {
		tensorBackend := newMetalTensorBackendForTest(t)

		shape, err := computetensor.NewShape([]int{0})
		So(err, ShouldBeNil)

		tensor, err := tensorBackend.UploadFloat64(shape, []float64{})
		So(err, ShouldBeNil)

		Convey("Close should succeed without freeing a nil buffer", func() {
			So(tensor.Close(), ShouldBeNil)
			So(tensor.Close(), ShouldBeNil)
		})
	})
}

func BenchmarkTensorBackend_UploadFloat64(benchmark *testing.B) {
	tensorBackend, err := NewTensorBackend()

	if err != nil {
		benchmark.Skipf("skipping benchmark due to setup error: %v", err)
	}

	defer func() {
		_ = tensorBackend.Close()
	}()

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

func BenchmarkTensorBackend_DownloadFloat64(benchmark *testing.B) {
	benchmark.ReportAllocs()

	tensorBackend, err := NewTensorBackend()

	if err != nil {
		benchmark.Skipf("skipping benchmark due to setup error: %v", err)
	}

	defer func() {
		_ = tensorBackend.Close()
	}()

	shape, err := computetensor.NewShape([]int{3})

	if err != nil {
		benchmark.Fatal(err)
	}

	uploaded, err := tensorBackend.UploadFloat64(shape, []float64{1, 2, 3})

	if err != nil {
		benchmark.Fatal(err)
	}

	defer func() {
		_ = uploaded.Close()
	}()

	benchmark.ResetTimer()

	for iteration := 0; iteration < benchmark.N; iteration++ {
		_, err := tensorBackend.DownloadFloat64(uploaded)

		if err != nil {
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

	t.Cleanup(func() {
		_ = tensorBackend.Close()
	})

	return tensorBackend
}
