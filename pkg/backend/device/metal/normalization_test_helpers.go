package metal

import (
	"math"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

func encodeNormValuesAsDType(values []float32, storageDType dtype.DType) []byte {
	if storageDType == dtype.Float32 {
		return dtypeconvert.Float32ToBytes(values)
	}

	return encodeFloat32ValuesAsDType(values, storageDType)
}

func hostLayerNormExpectedBytes(
	testingObject testing.TB,
	rows int,
	cols int,
	storageDType dtype.DType,
) []byte {
	testingObject.Helper()

	inputBytes, scaleBytes, biasBytes := normDTypeBytes(rows, cols, storageDType)
	shape := mustShapeForTest(testingObject, []int{rows, cols})
	paramShape := mustShapeForTest(testingObject, []int{cols})
	input, _ := tensor.NewFromBytes(shape, storageDType, inputBytes)
	scale, _ := tensor.NewFromBytes(paramShape, storageDType, scaleBytes)
	bias, _ := tensor.NewFromBytes(paramShape, storageDType, biasBytes)
	out, _ := tensor.NewZeroed(shape, storageDType)
	defer closeBenchmarkTensors(input, scale, bias, out)

	kernel := lookupHostLayerNormKernel(testingObject, storageDType)
	if err := kernel.Run(input, scale, bias, out); err != nil {
		testingObject.Fatal(err)
	}

	_, bytes, err := out.RawBytes()
	if err != nil {
		testingObject.Fatal(err)
	}

	return bytes
}

func hostRMSNormExpectedBytes(
	testingObject testing.TB,
	rows int,
	cols int,
	storageDType dtype.DType,
) []byte {
	testingObject.Helper()

	inputBytes, scaleBytes, _ := normDTypeBytes(rows, cols, storageDType)
	shape := mustShapeForTest(testingObject, []int{rows, cols})
	paramShape := mustShapeForTest(testingObject, []int{cols})
	input, _ := tensor.NewFromBytes(shape, storageDType, inputBytes)
	scale, _ := tensor.NewFromBytes(paramShape, storageDType, scaleBytes)
	out, _ := tensor.NewZeroed(shape, storageDType)
	defer closeBenchmarkTensors(input, scale, out)

	kernel := lookupHostRMSNormKernel(testingObject, storageDType)
	if err := kernel.Run(input, scale, out); err != nil {
		testingObject.Fatal(err)
	}

	_, bytes, err := out.RawBytes()
	if err != nil {
		testingObject.Fatal(err)
	}

	return bytes
}

func lookupHostLayerNormKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("layernorm", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Host)
	if !ok {
		testingObject.Fatalf("missing Host %s layernorm kernel", storageDType.Name())
	}

	return kernel
}

func lookupHostRMSNormKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("rmsnorm", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Host)
	if !ok {
		testingObject.Fatalf("missing Host %s rmsnorm kernel", storageDType.Name())
	}

	return kernel
}

func assertNormalizationBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	expectedBytes []byte,
) {
	testingObject.Helper()

	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, input, storageDType, expectedBytes, 2)
		return
	}

	actualDType, actualBytes, err := backend.Download(input)
	if err != nil {
		testingObject.Fatalf("Download failed: %v", err)
	}

	if actualDType != storageDType {
		testingObject.Fatalf("download dtype mismatch: got %s want %s", actualDType, storageDType)
	}

	actualValues := decodeDTypeBytesToFloat32(actualBytes, storageDType)
	expectedValues := decodeDTypeBytesToFloat32(expectedBytes, storageDType)
	assertNormalizationFloat32WithinULP(
		testingObject,
		actualValues,
		expectedValues,
		normalizationFloat32MaxULP,
	)
}

func assertNormalizationFloat32WithinULP(
	testingObject testing.TB,
	actualValues []float32,
	expectedValues []float32,
	maxULP uint32,
) {
	testingObject.Helper()

	maxDistance, maxIndex := maxNormalizationFloat32ULPDistance(actualValues, expectedValues)
	if maxDistance <= maxULP {
		return
	}

	testingObject.Fatalf(
		"normalization float32 max ULP mismatch at %d: got %08x (%g), want %08x (%g), distance %d > %d",
		maxIndex,
		math.Float32bits(actualValues[maxIndex]),
		actualValues[maxIndex],
		math.Float32bits(expectedValues[maxIndex]),
		expectedValues[maxIndex],
		maxDistance,
		maxULP,
	)
}

func maxNormalizationFloat32ULPDistance(
	actualValues []float32,
	expectedValues []float32,
) (uint32, int) {
	var maxDistance uint32
	var maxIndex int

	for index := range actualValues {
		distance := float32ULPDistance(actualValues[index], expectedValues[index])
		if distance <= maxDistance {
			continue
		}

		maxDistance = distance
		maxIndex = index
	}

	return maxDistance, maxIndex
}
