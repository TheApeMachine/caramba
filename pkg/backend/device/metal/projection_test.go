package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

func TestKernelRegistry_MetalProjectionDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalProjectionDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalProjectionDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalProjectionDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, inner := range parityElementCounts {
		inner := inner

		testingObject.Run(testNameForElementCount(inner), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" projection tensors", testingObject, func() {
				runLinearParityCase(testingObject, backend, storageDType, inner)
				runFusedQKVParityCase(testingObject, backend, storageDType, inner)
				runLoRAMergeParityCase(testingObject, backend, storageDType, inner)
				runLoRAApplyParityCase(testingObject, backend, storageDType, inner)
			})
		})
	}
}

func runLinearParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	inner int,
) {
	batch, outDim := 7, 11
	inputBytes, weightBytes, biasBytes, expectedBytes :=
		linearDTypeBytes(batch, inner, outDim, storageDType)
	input, weight, bias, out := linearTensorsForTest(
		testingObject, backend, batch, inner, outDim, storageDType,
		inputBytes, weightBytes, biasBytes,
	)
	defer closeBenchmarkTensors(input, weight, bias, out)

	err := lookupLinearKernel(testingObject, storageDType).Run(input, weight, bias, out)
	convey.So(err, convey.ShouldBeNil)
	assertProjectionBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}

func runFusedQKVParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	inner int,
) {
	batch, outDim := 5, 7
	inputBytes, weightBytes, biasBytes, expected :=
		fusedQKVDTypeBytes(batch, inner, outDim, storageDType)
	input, weight, bias, query, key, value := fusedQKVTensorsForTest(
		testingObject, backend, batch, inner, outDim, storageDType,
		inputBytes, weightBytes, biasBytes,
	)
	defer closeBenchmarkTensors(input, weight, bias, query, key, value)

	err := lookupFusedQKVKernel(testingObject, storageDType).Run(
		input, weight, bias, query, key, value,
	)
	convey.So(err, convey.ShouldBeNil)
	assertProjectionBytesForTest(testingObject, backend, query, storageDType, expected.query)
	assertProjectionBytesForTest(testingObject, backend, key, storageDType, expected.key)
	assertProjectionBytesForTest(testingObject, backend, value, storageDType, expected.value)
}

func runLoRAMergeParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	inner int,
) {
	outDim, rank := 9, 4
	baseBytes, loraABytes, loraBBytes, expectedBytes :=
		loraMergeDTypeBytes(outDim, rank, inner, storageDType)
	base, loraA, loraB, out := loraMergeTensorsForTest(
		testingObject, backend, outDim, rank, inner, storageDType,
		baseBytes, loraABytes, loraBBytes,
	)
	defer closeBenchmarkTensors(base, loraA, loraB, out)

	err := lookupLoRAMergeKernel(testingObject, storageDType).Run(base, loraA, loraB, out)
	convey.So(err, convey.ShouldBeNil)
	assertProjectionBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}

func runLoRAApplyParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	inner int,
) {
	batch, outDim, rank := 3, 8, 4
	baseBytes, loraABytes, loraBBytes, inputBytes, expectedBytes :=
		loraApplyDTypeBytes(batch, outDim, rank, inner, storageDType)
	base, loraA, loraB, input, out := loraApplyTensorsForTest(
		testingObject, backend, batch, outDim, rank, inner, storageDType,
		baseBytes, loraABytes, loraBBytes, inputBytes,
	)
	defer closeBenchmarkTensors(base, loraA, loraB, input, out)

	err := lookupLoRAApplyKernel(testingObject, storageDType).Run(base, loraA, loraB, input, out)
	convey.So(err, convey.ShouldBeNil)
	assertProjectionBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}

func lookupLinearKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("linear", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s linear kernel", storageDType.Name())
	}

	return kernel
}

func lookupFusedQKVKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("fused_qkv", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, storageDType,
		},
		Outputs: []dtype.DType{
			storageDType, storageDType, storageDType,
		},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s fused_qkv kernel", storageDType.Name())
	}

	return kernel
}

func lookupLoRAMergeKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("lora_merge", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s lora_merge kernel", storageDType.Name())
	}

	return kernel
}

func lookupLoRAApplyKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("lora_apply", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType, storageDType, storageDType, storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s lora_apply kernel", storageDType.Name())
	}

	return kernel
}

func linearTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	batch int,
	inner int,
	outDim int,
	storageDType dtype.DType,
	inputBytes []byte,
	weightBytes []byte,
	biasBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	input := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{batch, inner}),
		storageDType, inputBytes,
	)
	weight := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{outDim, inner}),
		storageDType, weightBytes,
	)
	bias := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{outDim}),
		storageDType, biasBytes,
	)
	out := emptyTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{batch, outDim}),
		storageDType,
	)

	return input, weight, bias, out
}

func fusedQKVTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	batch int,
	inner int,
	outDim int,
	storageDType dtype.DType,
	inputBytes []byte,
	weightBytes []byte,
	biasBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	input, weight, bias, _ := linearTensorsForTest(
		testingObject, backend, batch, inner, 3*outDim, storageDType,
		inputBytes, weightBytes, biasBytes,
	)
	queryShape := mustShapeForTest(testingObject, []int{batch, outDim})
	query := emptyTensorForTest(testingObject, backend, queryShape, storageDType)
	key := emptyTensorForTest(testingObject, backend, queryShape, storageDType)
	value := emptyTensorForTest(testingObject, backend, queryShape, storageDType)

	return input, weight, bias, query, key, value
}

func loraMergeTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	outDim int,
	rank int,
	inner int,
	storageDType dtype.DType,
	baseBytes []byte,
	loraABytes []byte,
	loraBBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	base := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{outDim, inner}),
		storageDType, baseBytes,
	)
	loraA := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{outDim, rank}),
		storageDType, loraABytes,
	)
	loraB := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{rank, inner}),
		storageDType, loraBBytes,
	)
	out := emptyTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{outDim, inner}),
		storageDType,
	)

	return base, loraA, loraB, out
}

func loraApplyTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	batch int,
	outDim int,
	rank int,
	inner int,
	storageDType dtype.DType,
	baseBytes []byte,
	loraABytes []byte,
	loraBBytes []byte,
	inputBytes []byte,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	base := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{batch, outDim}),
		storageDType, baseBytes,
	)
	loraA := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{outDim, rank}),
		storageDType, loraABytes,
	)
	loraB := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{rank, inner}),
		storageDType, loraBBytes,
	)
	input := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{batch, inner}),
		storageDType, inputBytes,
	)
	out := emptyTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{batch, outDim}),
		storageDType,
	)

	return base, loraA, loraB, input, out
}

func linearDTypeBytes(
	batch int,
	inner int,
	outDim int,
	storageDType dtype.DType,
) ([]byte, []byte, []byte, []byte) {
	inputValues := projectionValues(batch*inner, 17, 32)
	weightValues := projectionValues(outDim*inner, 29, 64)
	biasValues := projectionValues(outDim, 11, 16)
	inputBytes := encodeProjectionValuesAsDType(inputValues, storageDType)
	weightBytes := encodeProjectionValuesAsDType(weightValues, storageDType)
	biasBytes := encodeProjectionValuesAsDType(biasValues, storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	weightStored := decodeDTypeBytesToFloat32(weightBytes, storageDType)
	biasStored := decodeDTypeBytesToFloat32(biasBytes, storageDType)
	expectedValues := linearExpectedValues(inputStored, weightStored, biasStored, batch, inner, outDim)

	return inputBytes, weightBytes, biasBytes, encodeProjectionValuesAsDType(
		expectedValues, storageDType,
	)
}

type fusedQKVExpectedBytes struct {
	query []byte
	key   []byte
	value []byte
}

func fusedQKVDTypeBytes(
	batch int,
	inner int,
	outDim int,
	storageDType dtype.DType,
) ([]byte, []byte, []byte, fusedQKVExpectedBytes) {
	inputValues := projectionValues(batch*inner, 19, 32)
	weightValues := projectionValues(3*outDim*inner, 31, 64)
	biasValues := projectionValues(3*outDim, 13, 16)
	inputBytes := encodeProjectionValuesAsDType(inputValues, storageDType)
	weightBytes := encodeProjectionValuesAsDType(weightValues, storageDType)
	biasBytes := encodeProjectionValuesAsDType(biasValues, storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	weightStored := decodeDTypeBytesToFloat32(weightBytes, storageDType)
	biasStored := decodeDTypeBytesToFloat32(biasBytes, storageDType)
	query, key, value := fusedQKVExpectedValues(
		inputStored, weightStored, biasStored, batch, inner, outDim,
	)

	return inputBytes, weightBytes, biasBytes, fusedQKVExpectedBytes{
		query: encodeProjectionValuesAsDType(query, storageDType),
		key:   encodeProjectionValuesAsDType(key, storageDType),
		value: encodeProjectionValuesAsDType(value, storageDType),
	}
}

func loraMergeDTypeBytes(
	outDim int,
	rank int,
	inner int,
	storageDType dtype.DType,
) ([]byte, []byte, []byte, []byte) {
	baseValues := projectionValues(outDim*inner, 23, 64)
	loraAValues := projectionValues(outDim*rank, 17, 128)
	loraBValues := projectionValues(rank*inner, 19, 128)
	baseBytes := encodeProjectionValuesAsDType(baseValues, storageDType)
	loraABytes := encodeProjectionValuesAsDType(loraAValues, storageDType)
	loraBBytes := encodeProjectionValuesAsDType(loraBValues, storageDType)
	baseStored := decodeDTypeBytesToFloat32(baseBytes, storageDType)
	loraAStored := decodeDTypeBytesToFloat32(loraABytes, storageDType)
	loraBStored := decodeDTypeBytesToFloat32(loraBBytes, storageDType)
	expectedValues := loraMergeExpectedValues(
		baseStored, loraAStored, loraBStored, outDim, rank, inner,
	)

	return baseBytes, loraABytes, loraBBytes, encodeProjectionValuesAsDType(
		expectedValues, storageDType,
	)
}

func loraApplyDTypeBytes(
	batch int,
	outDim int,
	rank int,
	inner int,
	storageDType dtype.DType,
) ([]byte, []byte, []byte, []byte, []byte) {
	baseValues := projectionValues(batch*outDim, 23, 64)
	loraAValues := projectionValues(outDim*rank, 17, 128)
	loraBValues := projectionValues(rank*inner, 19, 128)
	inputValues := projectionValues(batch*inner, 29, 64)
	baseBytes := encodeProjectionValuesAsDType(baseValues, storageDType)
	loraABytes := encodeProjectionValuesAsDType(loraAValues, storageDType)
	loraBBytes := encodeProjectionValuesAsDType(loraBValues, storageDType)
	inputBytes := encodeProjectionValuesAsDType(inputValues, storageDType)
	baseStored := decodeDTypeBytesToFloat32(baseBytes, storageDType)
	loraAStored := decodeDTypeBytesToFloat32(loraABytes, storageDType)
	loraBStored := decodeDTypeBytesToFloat32(loraBBytes, storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	expectedValues := loraApplyExpectedValues(
		baseStored, loraAStored, loraBStored, inputStored, batch, outDim, rank, inner,
	)

	return baseBytes, loraABytes, loraBBytes, inputBytes,
		encodeProjectionValuesAsDType(expectedValues, storageDType)
}

func projectionValues(elementCount int, modulus int, divisor float32) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*3+5, modulus, divisor)
	}

	return values
}

func linearExpectedValues(
	input []float32,
	weight []float32,
	bias []float32,
	batch int,
	inner int,
	outDim int,
) []float32 {
	out := make([]float32, batch*outDim)

	for batchIndex := range batch {
		for outIndex := range outDim {
			out[batchIndex*outDim+outIndex] = linearExpectedCell(
				input, weight, bias, batchIndex, inner, outIndex,
			)
		}
	}

	return out
}

func linearExpectedCell(
	input []float32,
	weight []float32,
	bias []float32,
	batchIndex int,
	inner int,
	outIndex int,
) float32 {
	accumulator := bias[outIndex]

	for innerIndex := range inner {
		accumulator += input[batchIndex*inner+innerIndex] * weight[outIndex*inner+innerIndex]
	}

	return accumulator
}

func fusedQKVExpectedValues(
	input []float32,
	weight []float32,
	bias []float32,
	batch int,
	inner int,
	outDim int,
) ([]float32, []float32, []float32) {
	query := make([]float32, batch*outDim)
	key := make([]float32, batch*outDim)
	value := make([]float32, batch*outDim)

	for batchIndex := range batch {
		for outIndex := range outDim {
			outputIndex := batchIndex*outDim + outIndex
			query[outputIndex] = linearExpectedCell(input, weight, bias, batchIndex, inner, outIndex)
			key[outputIndex] = fusedQKVExpectedCell(
				input, weight, bias, batchIndex, inner, outDim+outIndex, outDim+outIndex,
			)
			value[outputIndex] = fusedQKVExpectedCell(
				input, weight, bias, batchIndex, inner, 2*outDim+outIndex, 2*outDim+outIndex,
			)
		}
	}

	return query, key, value
}

func fusedQKVExpectedCell(
	input []float32,
	weight []float32,
	bias []float32,
	batchIndex int,
	inner int,
	weightOutIndex int,
	biasIndex int,
) float32 {
	accumulator := bias[biasIndex]

	for innerIndex := range inner {
		accumulator += input[batchIndex*inner+innerIndex] *
			weight[weightOutIndex*inner+innerIndex]
	}

	return accumulator
}

func loraMergeExpectedValues(
	base []float32,
	loraA []float32,
	loraB []float32,
	outDim int,
	rank int,
	inner int,
) []float32 {
	out := make([]float32, outDim*inner)

	for outIndex := range outDim {
		for innerIndex := range inner {
			outputIndex := outIndex*inner + innerIndex
			accumulator := base[outputIndex]

			for rankIndex := range rank {
				accumulator += loraA[outIndex*rank+rankIndex] *
					loraB[rankIndex*inner+innerIndex]
			}

			out[outputIndex] = accumulator
		}
	}

	return out
}

func loraApplyExpectedValues(
	base []float32,
	loraA []float32,
	loraB []float32,
	input []float32,
	batch int,
	outDim int,
	rank int,
	inner int,
) []float32 {
	scratch := loraApplyExpectedScratch(input, loraB, batch, rank, inner)
	out := make([]float32, batch*outDim)

	for batchIndex := range batch {
		for outIndex := range outDim {
			outputIndex := batchIndex*outDim + outIndex
			accumulator := base[outputIndex]

			for rankIndex := range rank {
				accumulator += loraA[outIndex*rank+rankIndex] * scratch[batchIndex*rank+rankIndex]
			}

			out[outputIndex] = accumulator
		}
	}

	return out
}

func loraApplyExpectedScratch(
	input []float32,
	loraB []float32,
	batch int,
	rank int,
	inner int,
) []float32 {
	scratch := make([]float32, batch*rank)

	for batchIndex := range batch {
		for rankIndex := range rank {
			var accumulator float32

			for innerIndex := range inner {
				accumulator += input[batchIndex*inner+innerIndex] *
					loraB[rankIndex*inner+innerIndex]
			}

			scratch[batchIndex*rank+rankIndex] = accumulator
		}
	}

	return scratch
}

func encodeProjectionValuesAsDType(values []float32, storageDType dtype.DType) []byte {
	if storageDType == dtype.Float32 {
		return dtypeconvert.Float32ToBytes(values)
	}

	return encodeFloat32ValuesAsDType(values, storageDType)
}

func assertProjectionBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	expectedBytes []byte,
) {
	testingObject.Helper()

	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, input, storageDType, expectedBytes, 1)
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
	assertFloat32WithinULP(testingObject, actualValues, expectedValues, 1)
}
