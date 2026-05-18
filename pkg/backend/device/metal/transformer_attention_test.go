package metal

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestKernelRegistry_MetalAttentionAndRoPEDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalTransformerDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalAttentionAndRoPEDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalAttentionAndRoPEDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" transformer tensors", testingObject, func() {
				runAttentionParityCase(testingObject, backend, storageDType, elementCount)
				runRoPEParityCase(testingObject, backend, storageDType, elementCount)
			})
		})
	}
}

func runAttentionParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	depth int,
) {
	seqQ, seqK, valueDim := 5, 7, 11
	fixture := attentionFixtureForTest(seqQ, seqK, depth, valueDim, storageDType)
	query, key, value, out := attentionTensorsForTest(
		testingObject, backend, seqQ, seqK, depth, valueDim, storageDType, fixture,
	)
	defer closeBenchmarkTensors(query, key, value, out)

	err := lookupAttentionKernel(testingObject, storageDType).Run(query, key, value, out)
	convey.So(err, convey.ShouldBeNil)
	assertAttentionBytesForTest(testingObject, backend, out, storageDType, fixture)
}

func runRoPEParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	seqLen int,
) {
	numHeads, headDim := 3, 8
	fixture := ropeFixtureForTest(seqLen, numHeads, headDim, storageDType)
	input, out := ropeTensorsForTest(
		testingObject, backend, seqLen, numHeads, headDim, storageDType, fixture.inputBytes,
	)
	defer closeBenchmarkTensors(input, out)

	err := lookupRoPEKernel(testingObject, storageDType).Run(input, out)
	convey.So(err, convey.ShouldBeNil)
	assertRoPEBytesForTest(testingObject, backend, out, storageDType, fixture)
}

func lookupAttentionKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("attention", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType,
			storageDType,
			storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s attention kernel", storageDType.Name())
	}

	return kernel
}

func lookupRoPEKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("rope", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s rope kernel", storageDType.Name())
	}

	return kernel
}

type attentionFixture struct {
	queryBytes      []byte
	keyBytes        []byte
	valueBytes      []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

type ropeFixture struct {
	inputBytes      []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

func attentionTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	seqQ int,
	seqK int,
	depth int,
	valueDim int,
	storageDType dtype.DType,
	fixture attentionFixture,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	query := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{seqQ, depth}),
		storageDType, fixture.queryBytes,
	)
	key := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{seqK, depth}),
		storageDType, fixture.keyBytes,
	)
	value := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{seqK, valueDim}),
		storageDType, fixture.valueBytes,
	)
	out := emptyTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{seqQ, valueDim}),
		storageDType,
	)

	return query, key, value, out
}

func ropeTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	seqLen int,
	numHeads int,
	headDim int,
	storageDType dtype.DType,
	inputBytes []byte,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	shape := mustShapeForTest(testingObject, []int{seqLen, numHeads, headDim})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, inputBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	return input, out
}

func attentionFixtureForTest(
	seqQ int,
	seqK int,
	depth int,
	valueDim int,
	storageDType dtype.DType,
) attentionFixture {
	queryBytes := encodeLossValuesAsDType(attentionValues(seqQ*depth, 3), storageDType)
	keyBytes := encodeLossValuesAsDType(attentionValues(seqK*depth, 5), storageDType)
	valueBytes := encodeLossValuesAsDType(attentionValues(seqK*valueDim, 7), storageDType)
	queryStored := decodeDTypeBytesToFloat32(queryBytes, storageDType)
	keyStored := decodeDTypeBytesToFloat32(keyBytes, storageDType)
	valueStored := decodeDTypeBytesToFloat32(valueBytes, storageDType)
	expected := attentionExpected(queryStored, keyStored, valueStored, seqQ, seqK, depth, valueDim)

	return attentionFixture{
		queryBytes:      queryBytes,
		keyBytes:        keyBytes,
		valueBytes:      valueBytes,
		expectedBytes:   encodeLossValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}
}

func ropeFixtureForTest(
	seqLen int,
	numHeads int,
	headDim int,
	storageDType dtype.DType,
) ropeFixture {
	inputBytes := encodeLossValuesAsDType(attentionValues(seqLen*numHeads*headDim, 11), storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	expected := ropeExpected(inputStored, seqLen, numHeads, headDim)

	return ropeFixture{
		inputBytes:      inputBytes,
		expectedBytes:   encodeLossValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}
}

func attentionValues(elementCount int, salt int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*salt+13, 67, 32)
	}

	return values
}

func attentionExpected(
	query []float32,
	key []float32,
	value []float32,
	seqQ int,
	seqK int,
	depth int,
	valueDim int,
) []float32 {
	scores := attentionScoresExpected(query, key, seqQ, seqK, depth)
	attentionSoftmaxExpected(scores, seqQ, seqK)
	return attentionWeightedExpected(scores, value, seqQ, seqK, valueDim)
}

func attentionScoresExpected(
	query []float32,
	key []float32,
	seqQ int,
	seqK int,
	depth int,
) []float32 {
	scores := make([]float32, seqQ*seqK)
	scale := float32(1.0 / math.Sqrt(float64(depth)))

	for rowIndex := range seqQ {
		for colIndex := range seqK {
			scores[rowIndex*seqK+colIndex] = attentionScoreCell(
				query, key, rowIndex, colIndex, depth,
			) * scale
		}
	}

	return scores
}

func attentionScoreCell(
	query []float32,
	key []float32,
	rowIndex int,
	colIndex int,
	depth int,
) float32 {
	accumulator := float32(0)

	for depthIndex := range depth {
		accumulator += query[rowIndex*depth+depthIndex] * key[colIndex*depth+depthIndex]
	}

	return accumulator
}

func attentionSoftmaxExpected(scores []float32, seqQ int, seqK int) {
	for rowIndex := range seqQ {
		rowOffset := rowIndex * seqK
		maximum := logSumExpRowMaximum(scores, rowOffset, seqK)
		sum := logSumExpRowSum(scores, rowOffset, seqK, maximum)

		for colIndex := range seqK {
			scores[rowOffset+colIndex] =
				float32(math.Exp(float64(scores[rowOffset+colIndex]-maximum))) / sum
		}
	}
}

func attentionWeightedExpected(
	scores []float32,
	value []float32,
	seqQ int,
	seqK int,
	valueDim int,
) []float32 {
	out := make([]float32, seqQ*valueDim)

	for rowIndex := range seqQ {
		for colIndex := range valueDim {
			out[rowIndex*valueDim+colIndex] =
				attentionWeightedCell(scores, value, rowIndex, colIndex, seqK, valueDim)
		}
	}

	return out
}

func attentionWeightedCell(
	scores []float32,
	value []float32,
	rowIndex int,
	colIndex int,
	seqK int,
	valueDim int,
) float32 {
	accumulator := float32(0)

	for keyIndex := range seqK {
		accumulator += scores[rowIndex*seqK+keyIndex] * value[keyIndex*valueDim+colIndex]
	}

	return accumulator
}

func ropeExpected(input []float32, seqLen int, numHeads int, headDim int) []float32 {
	out := make([]float32, len(input))
	halfDim := headDim / 2

	for seqIndex := range seqLen {
		for headIndex := range numHeads {
			rowOffset := (seqIndex*numHeads + headIndex) * headDim

			for pairIndex := range halfDim {
				ropeExpectedPair(input, out, rowOffset, seqIndex, pairIndex, headDim)
			}
		}
	}

	return out
}

func ropeExpectedPair(
	input []float32,
	out []float32,
	rowOffset int,
	seqIndex int,
	pairIndex int,
	headDim int,
) {
	exponent := -float64(2*pairIndex) / float64(headDim)
	theta := float32(float64(float32(seqIndex)) * math.Pow(10000.0, exponent))
	cosTheta := float32(math.Cos(float64(theta)))
	sinTheta := float32(math.Sin(float64(theta)))
	even := input[rowOffset+2*pairIndex]
	odd := input[rowOffset+2*pairIndex+1]

	out[rowOffset+2*pairIndex] = even*cosTheta - odd*sinTheta
	out[rowOffset+2*pairIndex+1] = even*sinTheta + odd*cosTheta
}

func assertAttentionBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	fixture attentionFixture,
) {
	testingObject.Helper()

	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, input, storageDType, fixture.expectedBytes, 2)
		return
	}

	assertFloat32TensorForTest(testingObject, backend, input, fixture.expectedFloat32, 256)
}

func assertRoPEBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	fixture ropeFixture,
) {
	testingObject.Helper()

	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, input, storageDType, fixture.expectedBytes, 2)
		return
	}

	assertFloat32TensorForTest(testingObject, backend, input, fixture.expectedFloat32, 1024)
}
