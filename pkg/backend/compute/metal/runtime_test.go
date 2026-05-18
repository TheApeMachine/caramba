//go:build darwin && cgo

package metal

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalRuntime_NewFloat32Tensor(test *testing.T) {
	Convey("Given a Metal runtime allocator", test, func() {
		runtime, err := NewMetalRuntime(DefaultMetalRuntimeConfig())
		So(err, ShouldBeNil)
		defer func() {
			So(runtime.Close(), ShouldBeNil)
		}()

		shape, err := computetensor.NewShape([]int{1024})
		So(err, ShouldBeNil)

		Convey("It should allocate private resident tensors and reuse released buffers", func() {
			first, err := runtime.NewFloat32Tensor(shape, MetalAllocationTensor)
			So(err, ShouldBeNil)
			So(first.StorageMode(), ShouldEqual, MetalStorageModePrivate)
			So(first.Metadata().DType, ShouldEqual, computetensor.Float32)
			So(first.Metadata().Allocation, ShouldEqual, MetalAllocationTensor)

			beforeClose := runtime.Metrics()
			So(beforeClose.LiveBytes, ShouldEqual, int64(4096))
			So(beforeClose.PeakBytes, ShouldEqual, int64(4096))

			So(first.Close(), ShouldBeNil)

			afterClose := runtime.Metrics()
			So(afterClose.LiveBytes, ShouldEqual, int64(0))
			So(afterClose.PooledBytes, ShouldEqual, int64(4096))

			second, err := runtime.NewFloat32Tensor(shape, MetalAllocationScratch)
			So(err, ShouldBeNil)
			defer func() {
				So(second.Close(), ShouldBeNil)
			}()

			afterReuse := runtime.Metrics()
			So(afterReuse.ReusedBytes, ShouldEqual, int64(4096))
			So(afterReuse.PooledBytes, ShouldEqual, int64(0))
			So(second.Metadata().Allocation, ShouldEqual, MetalAllocationScratch)
		})
	})
}

func TestMetalRuntime_UploadFloat64(test *testing.T) {
	Convey("Given a Metal runtime allocator with pooled buffers", test, func() {
		runtime, err := NewMetalRuntime(DefaultMetalRuntimeConfig())
		So(err, ShouldBeNil)
		defer func() {
			So(runtime.Close(), ShouldBeNil)
		}()

		Convey("It should not pool undersized upload buffers by size class", func() {
			smallShape, err := computetensor.NewShape([]int{1})
			So(err, ShouldBeNil)

			upload, err := runtime.UploadFloat64(smallShape, []float64{1})
			So(err, ShouldBeNil)
			So(upload.Close(), ShouldBeNil)
			So(runtime.Metrics().PooledBytes, ShouldEqual, int64(0))

			largeShape, err := computetensor.NewShape([]int{49})
			So(err, ShouldBeNil)

			output, err := runtime.NewFloat32Tensor(largeShape, MetalAllocationTensor)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			So(runtime.Metrics().ReusedBytes, ShouldEqual, int64(0))
			_, err = tensorFloat64Values(output)
			So(err, ShouldBeNil)
		})
	})
}

func BenchmarkMetalRuntime_NewFloat32Tensor(benchmark *testing.B) {
	runtime, err := NewMetalRuntime(DefaultMetalRuntimeConfig())
	if err != nil {
		benchmark.Fatalf("new runtime: %v", err)
	}
	defer func() {
		if err := runtime.Close(); err != nil {
			benchmark.Fatalf("close runtime: %v", err)
		}
	}()

	shape, err := computetensor.NewShape([]int{1024})
	if err != nil {
		benchmark.Fatalf("shape: %v", err)
	}

	for benchmark.Loop() {
		output, err := runtime.NewFloat32Tensor(shape, MetalAllocationTensor)
		if err != nil {
			benchmark.Fatalf("allocate: %v", err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatalf("close tensor: %v", err)
		}
	}
}
