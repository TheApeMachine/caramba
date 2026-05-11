//go:build cgo && xla

package xla

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestNewTensorBackend(t *testing.T) {
	Convey("Given a PJRT runtime is available", t, func() {
		tensorBackend := newXLATensorBackendForTest(t)
		defer tensorBackend.Close()

		Convey("It should initialize resident XLA storage", func() {
			So(tensorBackend.Location(), ShouldEqual, computetensor.XLA)
		})
	})
}

func TestTensorBackend_UploadFloat64(t *testing.T) {
	Convey("Given a resident XLA tensor backend", t, func() {
		tensorBackend := newXLATensorBackendForTest(t)
		defer tensorBackend.Close()

		shape := mustXLAShape(t, []int{3})

		Convey("It should upload and download through PJRT buffers", func() {
			uploaded, err := tensorBackend.UploadFloat64(shape, []float64{1, 2, 3})
			So(err, ShouldBeNil)
			defer uploaded.Close()

			values, err := tensorBackend.DownloadFloat64(uploaded)
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2, 3})
		})
	})
}

func TestTensorBackend_Add(t *testing.T) {
	Convey("Given two resident XLA tensors", t, func() {
		tensorBackend := newXLATensorBackendForTest(t)
		defer tensorBackend.Close()

		left := uploadXLATensor(t, tensorBackend, []int{3}, []float64{1, 2, 3})
		defer left.Close()
		right := uploadXLATensor(t, tensorBackend, []int{3}, []float64{4, 5, 6})
		defer right.Close()

		Convey("It should add without leaving resident storage", func() {
			output, err := tensorBackend.Add(left, right)
			So(err, ShouldBeNil)
			defer output.Close()

			values, err := tensorBackend.DownloadFloat64(output)
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{5, 7, 9})
		})
	})
}

func TestTensorBackend_Matmul(t *testing.T) {
	Convey("Given resident XLA matrices", t, func() {
		tensorBackend := newXLATensorBackendForTest(t)
		defer tensorBackend.Close()

		left := uploadXLATensor(t, tensorBackend, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
		defer left.Close()
		right := uploadXLATensor(t, tensorBackend, []int{3, 2}, []float64{7, 8, 9, 10, 11, 12})
		defer right.Close()

		Convey("It should multiply matrices through PJRT", func() {
			output, err := tensorBackend.Matmul(left, right)
			So(err, ShouldBeNil)
			defer output.Close()

			values, err := tensorBackend.DownloadFloat64(output)
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{58, 64, 139, 154})
		})
	})
}

func TestTensorBackend_MatmulAddGELU(t *testing.T) {
	Convey("Given resident XLA matrices and bias", t, func() {
		tensorBackend := newXLATensorBackendForTest(t)
		defer tensorBackend.Close()

		left := uploadXLATensor(t, tensorBackend, []int{1, 2}, []float64{1, 2})
		defer left.Close()
		right := uploadXLATensor(t, tensorBackend, []int{2, 2}, []float64{3, 4, 5, 6})
		defer right.Close()
		bias := uploadXLATensor(t, tensorBackend, []int{2}, []float64{0.5, -0.5})
		defer bias.Close()

		Convey("It should fuse matmul, bias, and GELU", func() {
			output, err := tensorBackend.MatmulAddGELU(left, right, bias)
			So(err, ShouldBeNil)
			defer output.Close()

			values, err := tensorBackend.DownloadFloat64(output)
			So(err, ShouldBeNil)
			So(values[0], ShouldAlmostEqual, geluApprox(13.5), 1e-9)
			So(values[1], ShouldAlmostEqual, geluApprox(15.5), 1e-9)
		})
	})
}

func BenchmarkTensorBackend_MatmulAddGELU(benchmark *testing.B) {
	tensorBackend := newXLATensorBackendForBenchmark(benchmark)
	defer tensorBackend.Close()

	left := uploadXLATensorForBenchmark(benchmark, tensorBackend, []int{64, 128}, make([]float64, 64*128))
	defer left.Close()
	right := uploadXLATensorForBenchmark(benchmark, tensorBackend, []int{128, 64}, make([]float64, 128*64))
	defer right.Close()
	bias := uploadXLATensorForBenchmark(benchmark, tensorBackend, []int{64}, make([]float64, 64))
	defer bias.Close()

	benchmark.ResetTimer()

	for range benchmark.N {
		output, err := tensorBackend.MatmulAddGELU(left, right, bias)

		if err != nil {
			benchmark.Fatal(err)
		}

		_ = output.Close()
	}
}

func newXLATensorBackendForTest(t *testing.T) *TensorBackend {
	t.Helper()

	platform := xlaTensorPlatform(t)
	tensorBackend, err := NewTensorBackend(platform)

	if err != nil {
		t.Skip(err)
	}

	return tensorBackend
}

func newXLATensorBackendForBenchmark(benchmark *testing.B) *TensorBackend {
	benchmark.Helper()

	platform := xlaTensorPlatform(benchmark)
	tensorBackend, err := NewTensorBackend(platform)

	if err != nil {
		benchmark.Skip(err)
	}

	return tensorBackend
}

func xlaTensorPlatform(tb testing.TB) string {
	tb.Helper()

	for _, platform := range []string{"cpu", "gpu"} {
		config, err := NewPJRTConfig(platform)

		if err != nil {
			continue
		}

		if config.ValidateRuntime() == nil {
			return platform
		}
	}

	tb.Skip("xla tensor: no PJRT plugin configured")

	return ""
}

func uploadXLATensor(
	t *testing.T, tensorBackend *TensorBackend, dims []int, values []float64,
) computetensor.Float64Tensor {
	t.Helper()

	shape := mustXLAShape(t, dims)
	input, err := tensorBackend.UploadFloat64(shape, values)

	if err != nil {
		t.Fatal(err)
	}

	return input
}

func uploadXLATensorForBenchmark(
	benchmark *testing.B, tensorBackend *TensorBackend, dims []int, values []float64,
) computetensor.Float64Tensor {
	benchmark.Helper()

	shape, err := computetensor.NewShape(dims)

	if err != nil {
		benchmark.Fatal(err)
	}

	input, err := tensorBackend.UploadFloat64(shape, values)

	if err != nil {
		benchmark.Fatal(err)
	}

	return input
}

func mustXLAShape(t *testing.T, dims []int) computetensor.Shape {
	t.Helper()

	shape, err := computetensor.NewShape(dims)

	if err != nil {
		t.Fatalf("NewShape(%v): %v", dims, err)
	}

	return shape
}

func geluApprox(value float64) float64 {
	return 0.5 * value * (1 + math.Tanh(0.7978845608028654*(value+0.044715*value*value*value)))
}
