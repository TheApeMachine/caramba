//go:build darwin && cgo

package metal

import (
	"context"
	"strconv"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/qpool"
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

			metalTensor, ok := uploaded.(*Tensor)
			So(ok, ShouldBeTrue)
			So(metalTensor.StorageMode(), ShouldEqual, MetalStorageModePrivate)
			So(metalTensor.Layout(), ShouldEqual, MetalLayoutContiguous)
			So(metalTensor.Strides(), ShouldResemble, []int{1})
		})
	})
}

func TestRunner_ExecuteNoReadbackBeforeBoundary(t *testing.T) {
	Convey("Given a Metal graph execution", t, func() {
		tensorBackend := newMetalTensorBackendForTest(t)
		runner := NewRunnerWithBackend(tensorBackend)
		shape, err := computetensor.NewShape([]int{2})
		So(err, ShouldBeNil)

		input := ir.NewNode("input", ir.OpInput, shape)
		input.SetMetadata("values", []float64{-1, 2})
		output := ir.NewNode("relu", ir.OpReLU, shape)
		output.AddInput(input)

		graph := ir.NewGraph()
		graph.AddNode(input)
		graph.AddNode(output)

		before := tensorBackend.runtime.Metrics()
		results, err := runner.Execute(context.Background(), graph, []*ir.Node{output})
		after := tensorBackend.runtime.Metrics()

		Convey("It should keep intermediate and output tensors resident", func() {
			So(err, ShouldBeNil)
			So(results, ShouldHaveLength, 1)
			So(results["relu"].Location(), ShouldEqual, computetensor.Metal)
			So(after.TransferBytes-before.TransferBytes, ShouldEqual, int64(shape.Len()*4))
			So(results["relu"].Close(), ShouldBeNil)
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

func TestTensorBackend_Add(t *testing.T) {
	Convey("Given resident Metal tensors", t, func() {
		tensorBackend := newMetalTensorBackendForTest(t)

		shape, err := computetensor.NewShape([]int{2})
		So(err, ShouldBeNil)

		left, err := tensorBackend.UploadFloat64(shape, []float64{1, 2})
		So(err, ShouldBeNil)
		defer func() {
			So(left.Close(), ShouldBeNil)
		}()

		right, err := tensorBackend.UploadFloat64(shape, []float64{3, 4})
		So(err, ShouldBeNil)
		defer func() {
			So(right.Close(), ShouldBeNil)
		}()

		Convey("It should initialize math kernels before launching add", func() {
			output, err := tensorBackend.Add(left, right)

			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)

			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{4, 6})
		})
	})
}

func TestTensorBackend_GELU(t *testing.T) {
	Convey("Given a resident Metal tensor", t, func() {
		tensorBackend := newMetalTensorBackendForTest(t)

		shape, err := computetensor.NewShape([]int{1})
		So(err, ShouldBeNil)

		input, err := tensorBackend.UploadFloat64(shape, []float64{0})
		So(err, ShouldBeNil)
		defer func() {
			So(input.Close(), ShouldBeNil)
		}()

		Convey("It should initialize activation kernels before launching GELU", func() {
			output, err := tensorBackend.GELU(input)

			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)

			So(err, ShouldBeNil)
			So(values[0], ShouldAlmostEqual, 0, 1e-6)
		})
	})
}

func TestTensorBackend_CloseConcurrent(t *testing.T) {
	Convey("Given a Metal tensor backend", t, func() {
		tensorBackend := newMetalTensorBackendForTest(t)

		shape, err := computetensor.NewShape([]int{2})
		So(err, ShouldBeNil)

		Convey("Concurrent Close should be safe", func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			pool := qpool.NewQ(ctx, 8, 8, &qpool.Config{
				SchedulingTimeout:  time.Second,
				JobChannelCapacity: 8,
				Scaler:             nil,
			})
			defer pool.Close()

			results := make([]chan *qpool.QValue, 0, 8)

			for index := range 8 {
				results = append(results, pool.Schedule(
					metalCloseJobID(index),
					func(context.Context) (any, error) {
						_ = tensorBackend.Close()

						return nil, nil
					},
					qpool.WithExecTimeout(time.Second),
				))
			}

			for _, result := range results {
				So((<-result).Error, ShouldBeNil)
			}

			_, err := tensorBackend.UploadFloat64(shape, []float64{1, 2})
			So(err, ShouldNotBeNil)
		})
	})
}

func metalCloseJobID(index int) string {
	return "metal.tensor.close." + strconv.Itoa(index)
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

func BenchmarkTensorBackend_Add(benchmark *testing.B) {
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

	left, err := tensorBackend.UploadFloat64(shape, make([]float64, shape.Len()))

	if err != nil {
		benchmark.Fatal(err)
	}

	defer func() {
		_ = left.Close()
	}()

	right, err := tensorBackend.UploadFloat64(shape, make([]float64, shape.Len()))

	if err != nil {
		benchmark.Fatal(err)
	}

	defer func() {
		_ = right.Close()
	}()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := tensorBackend.Add(left, right)

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
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
