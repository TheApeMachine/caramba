package metal

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

var parityElementCounts = []int{1, 7, 64, 1024, 8192}

func TestNewBackend(t *testing.T) {
	convey.Convey("Given the Metal backend constructor", t, func() {
		backend, err := NewBackend()

		if err != nil {
			convey.So(errors.Is(err, tensor.ErrNeedsPlatformSetup), convey.ShouldBeTrue)

			return
		}

		defer func() {
			convey.So(backend.Close(), convey.ShouldBeNil)
		}()

		convey.So(backend.Location(), convey.ShouldEqual, tensor.Metal)
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

func TestBackend_UploadDownloadFloat32(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given a Metal float32 tensor upload", t, func() {
		shape, err := tensor.NewShape([]int{4})
		convey.So(err, convey.ShouldBeNil)

		values := []float32{1, -2, 3.5, 4.25}
		uploaded, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(values))
		convey.So(err, convey.ShouldBeNil)
		defer func() {
			convey.So(uploaded.Close(), convey.ShouldBeNil)
		}()

		sourceDType, bytes, err := backend.Download(uploaded)
		convey.So(err, convey.ShouldBeNil)
		convey.So(sourceDType, convey.ShouldEqual, dtype.Float32)

		actual, err := dtypeconvert.BytesToFloat32(sourceDType, bytes)
		convey.So(err, convey.ShouldBeNil)
		convey.So(actual, convey.ShouldResemble, values)
	})
}

func TestBackend_AddFloat32(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		t.Run(fmt.Sprintf("N=%d", elementCount), func(t *testing.T) {
			convey.Convey("Given two Metal float32 tensors", t, func() {
				shape, err := tensor.NewShape([]int{elementCount})
				convey.So(err, convey.ShouldBeNil)

				leftValues, rightValues, expectedValues := addFloat32ParityValues(elementCount)

				left, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(leftValues))
				convey.So(err, convey.ShouldBeNil)
				defer func() {
					convey.So(left.Close(), convey.ShouldBeNil)
				}()

				right, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(rightValues))
				convey.So(err, convey.ShouldBeNil)
				defer func() {
					convey.So(right.Close(), convey.ShouldBeNil)
				}()

				out, err := backend.AddFloat32(context.Background(), left, right)
				convey.So(err, convey.ShouldBeNil)
				defer func() {
					convey.So(out.Close(), convey.ShouldBeNil)
				}()

				actual := downloadFloat32ForTest(t, backend, out)
				convey.So(len(actual), convey.ShouldEqual, elementCount)
				assertFloat32BitwiseEqual(t, actual, expectedValues)
			})
		})
	}
}

func TestKernelRegistry_MetalAddFloat32(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given the device kernel registry", t, func() {
		kernel, ok := kernels.Default.LookupLocation("add", kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		}, tensor.Metal)
		convey.So(ok, convey.ShouldBeTrue)

		shape, err := tensor.NewShape([]int{1})
		convey.So(err, convey.ShouldBeNil)

		left, err := backend.Upload(
			shape,
			dtype.Float32,
			dtypeconvert.Float32ToBytes([]float32{2}),
		)
		convey.So(err, convey.ShouldBeNil)
		defer func() {
			convey.So(left.Close(), convey.ShouldBeNil)
		}()

		right, err := backend.Upload(
			shape,
			dtype.Float32,
			dtypeconvert.Float32ToBytes([]float32{3}),
		)
		convey.So(err, convey.ShouldBeNil)
		defer func() {
			convey.So(right.Close(), convey.ShouldBeNil)
		}()

		out, err := backend.bridge.empty(shape, dtype.Float32)
		convey.So(err, convey.ShouldBeNil)
		defer func() {
			convey.So(out.Close(), convey.ShouldBeNil)
		}()

		err = kernel.Run(left, right, out)
		convey.So(err, convey.ShouldBeNil)
		convey.So(downloadFloat32ForTest(t, backend, out), convey.ShouldResemble, []float32{5})
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

func BenchmarkNewBackend(b *testing.B) {
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

func BenchmarkBackend_AddFloat32(b *testing.B) {
	backend := newBackendForBenchmark(b)
	defer func() {
		_ = backend.Close()
	}()

	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		b.Run(fmt.Sprintf("N=%d", elementCount), func(b *testing.B) {
			benchmarkBackendAddFloat32(b, backend, elementCount)
		})
	}
}

func BenchmarkKernel_RunAddFloat32(b *testing.B) {
	backend := newBackendForBenchmark(b)
	defer func() {
		_ = backend.Close()
	}()

	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		b.Run(fmt.Sprintf("N=%d", elementCount), func(b *testing.B) {
			benchmarkKernelRunAddFloat32(b, backend, elementCount)
		})
	}
}

func benchmarkBackendAddFloat32(benchmark *testing.B, backend *Backend, elementCount int) {
	benchmark.Helper()

	shape, left, right := uploadAddFloat32BenchmarkInputs(benchmark, backend, elementCount)
	defer func() {
		_ = left.Close()
		_ = right.Close()
	}()

	benchmark.SetBytes(int64(shape.Len() * 3 * 4))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		out, err := backend.AddFloat32(context.Background(), left, right)
		if err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func benchmarkKernelRunAddFloat32(benchmark *testing.B, backend *Backend, elementCount int) {
	benchmark.Helper()

	shape, left, right := uploadAddFloat32BenchmarkInputs(benchmark, backend, elementCount)
	defer func() {
		_ = left.Close()
		_ = right.Close()
	}()

	out, err := backend.bridge.empty(shape, dtype.Float32)
	if err != nil {
		benchmark.Fatal(err)
	}
	defer func() {
		_ = out.Close()
	}()

	benchmark.SetBytes(int64(shape.Len() * 3 * 4))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := runMetalAddFloat32(left, right, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func uploadAddFloat32BenchmarkInputs(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
) (tensor.Shape, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	shape, err := tensor.NewShape([]int{elementCount})
	if err != nil {
		testingObject.Fatal(err)
	}

	leftValues, rightValues, _ := addFloat32ParityValues(elementCount)

	left, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(leftValues))
	if err != nil {
		testingObject.Fatal(err)
	}

	right, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(rightValues))
	if err != nil {
		_ = left.Close()
		testingObject.Fatal(err)
	}

	return shape, left, right
}

func addFloat32ParityValues(elementCount int) ([]float32, []float32, []float32) {
	leftValues := make([]float32, elementCount)
	rightValues := make([]float32, elementCount)
	expectedValues := make([]float32, elementCount)

	for index := range leftValues {
		leftValues[index] = float32((index%257)-128) * 0.125
		rightValues[index] = float32((index%131)-65) * 0.0625
		expectedValues[index] = leftValues[index] + rightValues[index]
	}

	return leftValues, rightValues, expectedValues
}

func assertFloat32BitwiseEqual(
	testingObject testing.TB,
	actualValues []float32,
	expectedValues []float32,
) {
	testingObject.Helper()

	if len(actualValues) != len(expectedValues) {
		testingObject.Fatalf("length mismatch: got %d want %d", len(actualValues), len(expectedValues))
	}

	for index := range actualValues {
		actualBits := math.Float32bits(actualValues[index])
		expectedBits := math.Float32bits(expectedValues[index])

		if actualBits != expectedBits {
			testingObject.Fatalf(
				"float32 bit mismatch at %d: got %08x (%g), want %08x (%g)",
				index,
				actualBits,
				actualValues[index],
				expectedBits,
				expectedValues[index],
			)
		}
	}
}

func newBackendForDeviceTest(testingObject testing.TB) *Backend {
	testingObject.Helper()

	backend, err := NewBackend()
	if errors.Is(err, tensor.ErrNeedsPlatformSetup) {
		testingObject.Skipf("Metal device unavailable: %v", err)
	}

	if err != nil {
		testingObject.Fatalf("NewBackend failed: %v", err)
	}

	return backend
}

func newBackendForBenchmark(benchmark *testing.B) *Backend {
	benchmark.Helper()

	backend, err := NewBackend()
	if errors.Is(err, tensor.ErrNeedsPlatformSetup) {
		benchmark.Skipf("Metal device unavailable: %v", err)
	}

	if err != nil {
		benchmark.Fatalf("NewBackend failed: %v", err)
	}

	return backend
}

func downloadFloat32ForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
) []float32 {
	testingObject.Helper()

	sourceDType, bytes, err := backend.Download(input)
	if err != nil {
		testingObject.Fatalf("Download failed: %v", err)
	}

	values, err := dtypeconvert.BytesToFloat32(sourceDType, bytes)
	if err != nil {
		testingObject.Fatalf("BytesToFloat32 failed: %v", err)
	}

	return values
}
