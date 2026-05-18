package metal

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestKernelRegistry_MetalAttentionVariantDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalTransformerDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalAttentionVariantDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalAttentionVariantDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, variant := range metalAttentionVariantCases() {
		variant := variant

		testingObject.Run(variant.name, func(testingObject *testing.T) {
			runMetalAttentionVariantDType(testingObject, backend, storageDType, variant)
		})
	}
}

func runMetalAttentionVariantDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
	variant metalAttentionVariantCase,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" "+variant.name+" tensors", testingObject, func() {
				runAttentionVariantParityCase(
					testingObject, backend, storageDType, variant, elementCount,
				)
			})
		})
	}
}

func runAttentionVariantParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	variant metalAttentionVariantCase,
	seqK int,
) {
	seqQ, numHeads, headDim := 3, 8, 64
	fixture := attentionVariantFixtureForTest(
		seqQ, seqK, numHeads, headDim, storageDType, variant,
	)
	query, key, value, out := attentionVariantTensorsForTest(
		testingObject, backend, seqQ, seqK, numHeads, headDim,
		storageDType, variant, fixture,
	)
	defer closeBenchmarkTensors(query, key, value, out)

	err := lookupAttentionVariantKernel(testingObject, storageDType, variant).Run(query, key, value, out)
	convey.So(err, convey.ShouldBeNil)
	assertAttentionVariantBytesForTest(testingObject, backend, out, storageDType, fixture)
}

func lookupAttentionVariantKernel(
	testingObject testing.TB,
	storageDType dtype.DType,
	variant metalAttentionVariantCase,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(variant.name, kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType,
			storageDType,
			storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s %s kernel", storageDType.Name(), variant.name)
	}

	return kernel
}

type metalAttentionVariantCase struct {
	name       string
	kvHeads    int
	causal     bool
	windowSize int
}

func metalAttentionVariantCases() []metalAttentionVariantCase {
	return []metalAttentionVariantCase{
		{name: "multi_head_attention", kvHeads: 8},
		{name: "grouped_query_attention", kvHeads: 2},
		{name: "sliding_window_attention", kvHeads: 8, causal: true, windowSize: 128},
	}
}

func attentionVariantTensorsForTest(
	testingObject testing.TB,
	backend *Backend,
	seqQ int,
	seqK int,
	numHeads int,
	headDim int,
	storageDType dtype.DType,
	variant metalAttentionVariantCase,
	fixture attentionFixture,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	queryFeatures := numHeads * headDim
	kvFeatures := variant.kvHeads * headDim
	query := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{seqQ, queryFeatures}),
		storageDType, fixture.queryBytes,
	)
	key := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{seqK, kvFeatures}),
		storageDType, fixture.keyBytes,
	)
	value := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{seqK, kvFeatures}),
		storageDType, fixture.valueBytes,
	)
	out := emptyTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{seqQ, queryFeatures}),
		storageDType,
	)

	return query, key, value, out
}

func attentionVariantFixtureForTest(
	seqQ int,
	seqK int,
	numHeads int,
	headDim int,
	storageDType dtype.DType,
	variant metalAttentionVariantCase,
) attentionFixture {
	queryFeatures := numHeads * headDim
	kvFeatures := variant.kvHeads * headDim
	queryBytes := encodeLossValuesAsDType(attentionValues(seqQ*queryFeatures, 17), storageDType)
	keyBytes := encodeLossValuesAsDType(attentionValues(seqK*kvFeatures, 19), storageDType)
	valueBytes := encodeLossValuesAsDType(attentionVariantValues(seqK*kvFeatures, 23), storageDType)
	queryStored := decodeDTypeBytesToFloat32(queryBytes, storageDType)
	keyStored := decodeDTypeBytesToFloat32(keyBytes, storageDType)
	valueStored := decodeDTypeBytesToFloat32(valueBytes, storageDType)
	expected := attentionVariantExpected(
		queryStored, keyStored, valueStored,
		seqQ, seqK, numHeads, variant.kvHeads, headDim, variant,
	)

	return attentionFixture{
		queryBytes:      queryBytes,
		keyBytes:        keyBytes,
		valueBytes:      valueBytes,
		expectedBytes:   encodeLossValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}
}

func attentionVariantValues(elementCount int, salt int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = 0.125 + float32((index*salt+7)%53)/128
	}

	return values
}

func attentionVariantExpected(
	query []float32,
	key []float32,
	value []float32,
	seqQ int,
	seqK int,
	numHeads int,
	kvHeads int,
	headDim int,
	variant metalAttentionVariantCase,
) []float32 {
	out := make([]float32, seqQ*numHeads*headDim)

	for rowIndex := range seqQ {
		for headIndex := range numHeads {
			for dimIndex := range headDim {
				out[(rowIndex*numHeads+headIndex)*headDim+dimIndex] =
					attentionVariantCell(
						query, key, value, rowIndex, headIndex, dimIndex,
						seqK, numHeads, kvHeads, headDim, variant,
					)
			}
		}
	}

	return out
}

func attentionVariantCell(
	query []float32,
	key []float32,
	value []float32,
	rowIndex int,
	headIndex int,
	dimIndex int,
	seqK int,
	numHeads int,
	kvHeads int,
	headDim int,
	variant metalAttentionVariantCase,
) float32 {
	maxScore := float32(math.Inf(-1))
	normalizer := float32(0)
	accumulator := float32(0)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	headsPerKVHead := numHeads / kvHeads
	kvHeadIndex := headIndex / headsPerKVHead

	for keyIndex := range seqK {
		if !attentionVariantKeepsKey(rowIndex, keyIndex, variant) {
			continue
		}

		score := attentionVariantScore(
			query, key, rowIndex, keyIndex, headIndex, kvHeadIndex, numHeads, kvHeads, headDim,
		) * scale
		oldMax := maxScore
		maxScore = max(maxScore, score)
		alpha := float32(math.Exp(float64(oldMax - maxScore)))
		shifted := float32(math.Exp(float64(score - maxScore)))
		normalizer = normalizer*alpha + shifted
		accumulator = accumulator*alpha + shifted*
			value[(keyIndex*kvHeads+kvHeadIndex)*headDim+dimIndex]
	}

	if normalizer == 0 {
		return 0
	}

	return accumulator / normalizer
}

func attentionVariantKeepsKey(
	rowIndex int,
	keyIndex int,
	variant metalAttentionVariantCase,
) bool {
	if variant.causal && keyIndex > rowIndex {
		return false
	}

	if variant.windowSize > 0 && rowIndex >= keyIndex && rowIndex-keyIndex >= variant.windowSize {
		return false
	}

	return true
}

func attentionVariantScore(
	query []float32,
	key []float32,
	rowIndex int,
	keyIndex int,
	headIndex int,
	kvHeadIndex int,
	numHeads int,
	kvHeads int,
	headDim int,
) float32 {
	accumulator := float32(0)
	queryOffset := (rowIndex*numHeads + headIndex) * headDim
	keyOffset := (keyIndex*kvHeads + kvHeadIndex) * headDim

	for depthIndex := range headDim {
		accumulator += query[queryOffset+depthIndex] * key[keyOffset+depthIndex]
	}

	return accumulator
}

func assertAttentionVariantBytesForTest(
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
