package metal

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestKernelRegistry_MetalAttentionAndFlashDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalTransformerDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalAttentionDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalAttentionDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" transformer tensors", testingObject, func() {
				runAttentionParityCase(testingObject, backend, storageDType, elementCount)
				runFlashAttentionParityCase(testingObject, backend, storageDType, elementCount)
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

func runFlashAttentionParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	depth int,
) {
	seqQ, seqK, valueDim := 5, 7, 11
	fixture := flashAttentionFixtureForTest(seqQ, seqK, depth, valueDim, storageDType)
	query, key, value, out := attentionTensorsForTest(
		testingObject, backend, seqQ, seqK, depth, valueDim, storageDType, fixture,
	)
	defer closeBenchmarkTensors(query, key, value, out)

	err := lookupFlashAttentionKernel(testingObject, storageDType).Run(query, key, value, out)
	convey.So(err, convey.ShouldBeNil)
	assertAttentionBytesForTest(testingObject, backend, out, storageDType, fixture)
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

func lookupFlashAttentionKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("flash_attention", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType,
			storageDType,
			storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s flash_attention kernel", storageDType.Name())
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

func flashAttentionFixtureForTest(
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
	expected := flashAttentionExpected(
		queryStored, keyStored, valueStored, seqQ, seqK, depth, valueDim,
	)

	return attentionFixture{
		queryBytes:      queryBytes,
		keyBytes:        keyBytes,
		valueBytes:      valueBytes,
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

func flashAttentionExpected(
	query []float32,
	key []float32,
	value []float32,
	seqQ int,
	seqK int,
	depth int,
	valueDim int,
) []float32 {
	out := make([]float32, seqQ*valueDim)

	for rowIndex := range seqQ {
		for colIndex := range valueDim {
			out[rowIndex*valueDim+colIndex] = flashAttentionCell(
				query, key, value, rowIndex, colIndex, seqK, depth, valueDim,
			)
		}
	}

	return out
}

func flashAttentionCell(
	query []float32,
	key []float32,
	value []float32,
	rowIndex int,
	colIndex int,
	seqK int,
	depth int,
	valueDim int,
) float32 {
	maxScore := float32(math.Inf(-1))
	normalizer := float32(0)
	accumulator := float32(0)
	scale := float32(1.0 / math.Sqrt(float64(depth)))

	for keyIndex := range seqK {
		score := attentionScoreCell(query, key, rowIndex, keyIndex, depth) * scale
		oldMax := maxScore
		maxScore = max(maxScore, score)
		alpha := float32(math.Exp(float64(oldMax - maxScore)))
		shifted := float32(math.Exp(float64(score - maxScore)))
		normalizer = normalizer*alpha + shifted
		accumulator = accumulator*alpha + shifted*value[keyIndex*valueDim+colIndex]
	}

	if normalizer == 0 {
		return 0
	}

	return accumulator / normalizer
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

	assertFloat32TensorForTest(testingObject, backend, input, fixture.expectedFloat32, 512)
}
