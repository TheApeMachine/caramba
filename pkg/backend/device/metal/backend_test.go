package metal

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
)

var parityElementCounts = []int{1, 7, 64, 1024, 8192}

type binaryFloat32Case struct {
	name      string
	operation metalBinaryFloat32Operation
	apply     func(*Backend, context.Context, tensor.Tensor, tensor.Tensor) (tensor.Tensor, error)
	maxULP    uint32
	dtypeULP  uint32
}

var binaryFloat32Cases = []binaryFloat32Case{
	{name: "add", operation: metalBinaryFloat32Add, apply: (*Backend).AddFloat32},
	{name: "sub", operation: metalBinaryFloat32Sub, apply: (*Backend).SubFloat32},
	{name: "mul", operation: metalBinaryFloat32Mul, apply: (*Backend).MulFloat32},
	{name: "div", operation: metalBinaryFloat32Div, apply: (*Backend).DivFloat32},
	{name: "max", operation: metalBinaryFloat32Max, apply: (*Backend).MaxFloat32},
	{name: "min", operation: metalBinaryFloat32Min, apply: (*Backend).MinFloat32},
	{name: "eq", operation: metalBinaryFloat32Eq, apply: (*Backend).EqFloat32},
	{name: "ne", operation: metalBinaryFloat32Ne, apply: (*Backend).NeFloat32},
	{name: "lt", operation: metalBinaryFloat32Lt, apply: (*Backend).LtFloat32},
	{name: "le", operation: metalBinaryFloat32Le, apply: (*Backend).LeFloat32},
	{name: "gt", operation: metalBinaryFloat32Gt, apply: (*Backend).GtFloat32},
	{name: "ge", operation: metalBinaryFloat32Ge, apply: (*Backend).GeFloat32},
	{name: "pow", operation: metalBinaryFloat32Pow, apply: (*Backend).PowFloat32, maxULP: 4, dtypeULP: 2},
	{name: "atan2", operation: metalBinaryFloat32Atan2, apply: (*Backend).Atan2Float32, maxULP: 8, dtypeULP: 2},
	{name: "mod", operation: metalBinaryFloat32Mod, apply: (*Backend).ModFloat32},
}

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
		convey.So(caps.SupportsAsync, convey.ShouldBeFalse)
	})
}

func TestBackend_Capabilities_Device(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given an opened Metal backend", t, func() {
		caps := backend.Capabilities()
		convey.So(caps.SupportsAsync, convey.ShouldBeTrue)
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

func TestBackend_UploadAsyncFloat32(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given an async Metal float32 tensor upload", t, func() {
		shape, err := tensor.NewShape([]int{8192})
		convey.So(err, convey.ShouldBeNil)

		values, _, _ := addFloat32ParityValues(shape.Len())
		uploaded, err := backend.UploadAsync(shape, dtype.Float32, dtypeconvert.Float32ToBytes(values))
		convey.So(err, convey.ShouldBeNil)
		defer func() {
			convey.So(uploaded.Close(), convey.ShouldBeNil)
		}()

		actual := downloadFloat32ForTest(t, backend, uploaded)
		assertFloat32BitwiseEqual(t, actual, values)
	})
}

func TestBackend_AddFloat32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[0])
}

func TestBackend_SubFloat32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[1])
}

func TestBackend_MulFloat32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[2])
}

func TestBackend_DivFloat32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[3])
}

func TestBackend_MaxFloat32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[4])
}

func TestBackend_MinFloat32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[5])
}

func TestBackend_EqFloat32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[6])
}

func TestBackend_NeFloat32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[7])
}

func TestBackend_LtFloat32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[8])
}

func TestBackend_LeFloat32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[9])
}

func TestBackend_GtFloat32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[10])
}

func TestBackend_GeFloat32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[11])
}

func TestBackend_PowFloat32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[12])
}

func TestBackend_Atan2Float32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[13])
}

func TestBackend_ModFloat32(t *testing.T) {
	testBackendBinaryFloat32(t, binaryFloat32Cases[14])
}

func testBackendBinaryFloat32(t *testing.T, testCase binaryFloat32Case) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		t.Run(fmt.Sprintf("N=%d", elementCount), func(t *testing.T) {
			convey.Convey("Given two Metal float32 tensors for "+testCase.name, t, func() {
				shape, err := tensor.NewShape([]int{elementCount})
				convey.So(err, convey.ShouldBeNil)

				leftValues, rightValues, expectedValues := binaryFloat32ParityValues(
					elementCount,
					testCase.name,
				)

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

				out, err := testCase.apply(backend, context.Background(), left, right)
				convey.So(err, convey.ShouldBeNil)
				defer func() {
					convey.So(out.Close(), convey.ShouldBeNil)
				}()

				actual := downloadFloat32ForTest(t, backend, out)
				convey.So(len(actual), convey.ShouldEqual, elementCount)
				assertBinaryFloat32Parity(t, actual, expectedValues, testCase.maxULP)
			})
		})
	}
}

func TestBackend_AddFloat32_CloseInputsBeforeDownload(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given a queued Metal add whose inputs are closed immediately", t, func() {
		shape, err := tensor.NewShape([]int{8192})
		convey.So(err, convey.ShouldBeNil)

		leftValues, rightValues, expectedValues := binaryFloat32ParityValues(shape.Len(), "add")
		left, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(leftValues))
		convey.So(err, convey.ShouldBeNil)

		right, err := backend.Upload(shape, dtype.Float32, dtypeconvert.Float32ToBytes(rightValues))
		convey.So(err, convey.ShouldBeNil)

		out, err := backend.AddFloat32(context.Background(), left, right)
		convey.So(err, convey.ShouldBeNil)
		defer func() {
			convey.So(out.Close(), convey.ShouldBeNil)
		}()

		convey.So(left.Close(), convey.ShouldBeNil)
		convey.So(right.Close(), convey.ShouldBeNil)
		assertFloat32BitwiseEqual(t, downloadFloat32ForTest(t, backend, out), expectedValues)
	})
}

func TestBackend_AddFloat32_CloseOutputBeforeCompletion(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given a queued Metal add whose output is closed immediately", t, func() {
		shape, err := tensor.NewShape([]int{8192})
		convey.So(err, convey.ShouldBeNil)

		leftValues, rightValues, _ := binaryFloat32ParityValues(shape.Len(), "add")
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
		convey.So(out.Close(), convey.ShouldBeNil)
		convey.So(out.State(), convey.ShouldEqual, tensor.StateClosed)
	})
}

func TestMetalBufferPool_AlignedBuckets(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	convey.Convey("Given closed Metal tensors with nearby byte sizes", t, func() {
		firstShape, err := tensor.NewShape([]int{1})
		convey.So(err, convey.ShouldBeNil)

		first, err := backend.Upload(
			firstShape,
			dtype.Float32,
			dtypeconvert.Float32ToBytes([]float32{1}),
		)
		convey.So(err, convey.ShouldBeNil)
		convey.So(first.Close(), convey.ShouldBeNil)

		secondShape, err := tensor.NewShape([]int{7})
		convey.So(err, convey.ShouldBeNil)

		second, err := backend.Upload(
			secondShape,
			dtype.Float32,
			dtypeconvert.Float32ToBytes([]float32{1, 2, 3, 4, 5, 6, 7}),
		)
		convey.So(err, convey.ShouldBeNil)
		convey.So(second.Close(), convey.ShouldBeNil)

		backend.bridge.pool.mutex.Lock()
		defer backend.bridge.pool.mutex.Unlock()

		convey.So(len(backend.bridge.pool.buffer[256]), convey.ShouldBeGreaterThanOrEqualTo, 1)
		_, hasFourByteBucket := backend.bridge.pool.buffer[4]
		_, hasTwentyEightByteBucket := backend.bridge.pool.buffer[28]
		convey.So(hasFourByteBucket, convey.ShouldBeFalse)
		convey.So(hasTwentyEightByteBucket, convey.ShouldBeFalse)
	})
}

func TestKernelRegistry_MetalBinaryFloat32(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	for _, testCase := range binaryFloat32Cases {
		testCase := testCase

		t.Run(testCase.name, func(t *testing.T) {
			convey.Convey("Given the device kernel registry for "+testCase.name, t, func() {
				kernel, ok := kernels.Default.LookupLocation(testCase.name, kernels.Signature{
					Layout:  tensor.LayoutDense,
					Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
					Outputs: []dtype.DType{dtype.Float32},
				}, tensor.Metal)
				convey.So(ok, convey.ShouldBeTrue)

				shape, err := tensor.NewShape([]int{1})
				convey.So(err, convey.ShouldBeNil)

				leftValues, rightValues, expectedValues := binaryFloat32ParityValues(1, testCase.name)
				left, err := backend.Upload(
					shape,
					dtype.Float32,
					dtypeconvert.Float32ToBytes(leftValues),
				)
				convey.So(err, convey.ShouldBeNil)
				defer func() {
					convey.So(left.Close(), convey.ShouldBeNil)
				}()

				right, err := backend.Upload(
					shape,
					dtype.Float32,
					dtypeconvert.Float32ToBytes(rightValues),
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
				assertBinaryFloat32Parity(
					t,
					downloadFloat32ForTest(t, backend, out),
					expectedValues,
					testCase.maxULP,
				)
			})
		})
	}
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

func BenchmarkBackend_BinaryFloat32(b *testing.B) {
	backend := newBackendForBenchmark(b)
	defer func() {
		_ = backend.Close()
	}()

	for _, testCase := range binaryFloat32Cases {
		testCase := testCase

		b.Run(testCase.name, func(b *testing.B) {
			for _, elementCount := range parityElementCounts {
				elementCount := elementCount

				b.Run(fmt.Sprintf("N=%d", elementCount), func(b *testing.B) {
					benchmarkBackendBinaryFloat32(b, backend, testCase, elementCount)
				})
			}
		})
	}
}

func BenchmarkKernel_RunBinaryFloat32(b *testing.B) {
	backend := newBackendForBenchmark(b)
	defer func() {
		_ = backend.Close()
	}()

	for _, testCase := range binaryFloat32Cases {
		testCase := testCase

		b.Run(testCase.name, func(b *testing.B) {
			for _, elementCount := range parityElementCounts {
				elementCount := elementCount

				b.Run(fmt.Sprintf("N=%d", elementCount), func(b *testing.B) {
					benchmarkKernelRunBinaryFloat32(b, backend, testCase, elementCount)
				})
			}
		})
	}
}

func benchmarkBackendBinaryFloat32(
	benchmark *testing.B,
	backend *Backend,
	testCase binaryFloat32Case,
	elementCount int,
) {
	benchmark.Helper()

	shape, left, right := uploadBinaryFloat32BenchmarkInputs(
		benchmark,
		backend,
		testCase.name,
		elementCount,
	)
	defer func() {
		_ = left.Close()
		_ = right.Close()
	}()

	benchmark.SetBytes(int64(shape.Len() * 3 * 4))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		out, err := testCase.apply(backend, context.Background(), left, right)
		if err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func benchmarkKernelRunBinaryFloat32(
	benchmark *testing.B,
	backend *Backend,
	testCase binaryFloat32Case,
	elementCount int,
) {
	benchmark.Helper()

	shape, left, right := uploadBinaryFloat32BenchmarkInputs(
		benchmark,
		backend,
		testCase.name,
		elementCount,
	)
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
		if err := runMetalBinaryFloat32(testCase.operation, left, right, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func uploadBinaryFloat32BenchmarkInputs(
	testingObject testing.TB,
	backend *Backend,
	name string,
	elementCount int,
) (tensor.Shape, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	shape, err := tensor.NewShape([]int{elementCount})
	if err != nil {
		testingObject.Fatal(err)
	}

	leftValues, rightValues, _ := binaryFloat32ParityValues(elementCount, name)

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
	return binaryFloat32ParityValues(elementCount, "add")
}

func binaryFloat32ParityValues(
	elementCount int,
	name string,
) ([]float32, []float32, []float32) {
	leftValues := make([]float32, elementCount)
	rightValues := make([]float32, elementCount)
	expectedValues := make([]float32, elementCount)

	for index := range leftValues {
		leftValues[index] = binaryFloat32LeftValue(index, name)
		rightValues[index] = binaryFloat32RightValue(index, name)

		if canUseEqualBinaryInputs(name) && index%13 == 0 && leftValues[index] != 0 {
			rightValues[index] = leftValues[index]
		}

		expectedValues[index] = binaryFloat32Expected(
			name,
			leftValues[index],
			rightValues[index],
		)
	}

	return leftValues, rightValues, expectedValues
}

func binaryFloat32LeftValue(index int, name string) float32 {
	switch name {
	case "pow":
		value := 1 << uint(2*(index%4))
		return float32(value)
	case "mod":
		return float32((index%511)-255) + 0.5
	}

	return float32((index % 511) - 255)
}

func binaryFloat32RightValue(index int, name string) float32 {
	switch name {
	case "pow":
		return binaryFloat32PowExponent(index)
	case "mod":
		value := 1 << uint(index%4)
		return float32(value)
	}

	integerValue := 1 << uint(index%4)
	value := float32(integerValue)

	if index%5 == 0 {
		return -value
	}

	return value
}

func binaryFloat32PowExponent(index int) float32 {
	values := []float32{-2, -1, -0.5, 0, 0.5, 1, 2, 3}
	return values[index%len(values)]
}

func canUseEqualBinaryInputs(name string) bool {
	switch name {
	case "div", "pow", "atan2", "mod":
		return false
	}

	return true
}

func binaryFloat32Expected(name string, left float32, right float32) float32 {
	switch name {
	case "add":
		return left + right
	case "sub":
		return left - right
	case "mul":
		return left * right
	case "div":
		return left / right
	case "max":
		if left > right {
			return left
		}

		return right
	case "min":
		if left < right {
			return left
		}

		return right
	case "eq":
		if left == right {
			return 1
		}

		return 0
	case "ne":
		if left != right {
			return 1
		}

		return 0
	case "lt":
		if left < right {
			return 1
		}

		return 0
	case "le":
		if left <= right {
			return 1
		}

		return 0
	case "gt":
		if left > right {
			return 1
		}

		return 0
	case "ge":
		if left >= right {
			return 1
		}

		return 0
	case "pow":
		return float32(math.Pow(float64(left), float64(right)))
	case "atan2":
		return float32(math.Atan2(float64(left), float64(right)))
	case "mod":
		return float32(math.Mod(float64(left), float64(right)))
	}

	panic("unknown binary float32 operation: " + name)
}

func assertBinaryFloat32Parity(
	testingObject testing.TB,
	actualValues []float32,
	expectedValues []float32,
	maxULP uint32,
) {
	testingObject.Helper()

	if maxULP == 0 {
		assertFloat32BitwiseEqual(testingObject, actualValues, expectedValues)
		return
	}

	assertFloat32WithinULP(testingObject, actualValues, expectedValues, maxULP)
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
