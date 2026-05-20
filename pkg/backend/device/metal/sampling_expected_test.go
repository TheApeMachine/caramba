package metal

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"sort"
	"testing"

	computekernels "github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type samplingFixture struct {
	inputBytes []byte
	expected   int32
}

func greedySamplingFixtureForTest(
	elementCount int,
	storageDType dtype.DType,
) samplingFixture {
	values := greedySamplingValues(elementCount)
	inputBytes := encodeLossValuesAsDType(values, storageDType)
	storedValues := decodeDTypeBytesToFloat32(inputBytes, storageDType)

	return samplingFixture{
		inputBytes: inputBytes,
		expected:   greedySamplingExpected(storedValues),
	}
}

func drawSamplingFixtureForTest(
	elementCount int,
	storageDType dtype.DType,
) samplingFixture {
	values := drawSamplingValues(elementCount)
	inputBytes := encodeLossValuesAsDType(values, storageDType)
	storedValues := decodeDTypeBytesToFloat32(inputBytes, storageDType)

	return samplingFixture{
		inputBytes: inputBytes,
		expected:   drawSamplingExpected(storedValues),
	}
}

func greedySamplingValues(elementCount int) []float32 {
	values := make([]float32, elementCount)
	winner := elementCount / 2

	for index := range values {
		values[index] = -float32((index%31)+1) / 16.0
	}

	values[winner] = 4.0

	return values
}

func drawSamplingValues(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = 0.5 - float32((index*7)%19)/32.0
	}

	return values
}

func greedySamplingExpected(values []float32) int32 {
	maxIndex := 0
	maxLogit := values[0]

	for index, value := range values[1:] {
		if value > maxLogit {
			maxLogit = value
			maxIndex = index + 1
		}
	}

	return int32(maxIndex)
}

func drawSamplingExpected(values []float32) int32 {
	probabilities, indices := samplingSoftmaxAndSort(values)
	target := samplingDefaultTargetForTest()
	cumulative := float32(0)

	for index, probability := range probabilities {
		cumulative += probability

		if cumulative >= target {
			return int32(indices[index])
		}
	}

	return int32(indices[len(indices)-1])
}

func samplingSoftmaxAndSort(values []float32) ([]float32, []int) {
	probabilities := make([]float32, len(values))
	indices := make([]int, len(values))
	maximum := values[0]

	for _, value := range values[1:] {
		if value > maximum {
			maximum = value
		}
	}

	denominator := float64(0)
	for index, value := range values {
		shifted := math.Exp(float64(value - maximum))
		probabilities[index] = float32(shifted)
		indices[index] = index
		denominator += shifted
	}

	scale := float32(1.0 / denominator)
	for index := range probabilities {
		probabilities[index] *= scale
	}

	sort.SliceStable(indices, func(left int, right int) bool {
		return probabilities[indices[left]] > probabilities[indices[right]]
	})

	sorted := make([]float32, len(probabilities))
	for resultIndex, originalIndex := range indices {
		sorted[resultIndex] = probabilities[originalIndex]
	}

	return sorted, indices
}

func samplingDefaultTargetForTest() float32 {
	config := computekernels.DefaultSamplingConfig()
	source := rand.NewChaCha8([32]byte{
		byte(config.Seed), byte(config.Seed >> 8), byte(config.Seed >> 16), byte(config.Seed >> 24),
		byte(config.Seed >> 32), byte(config.Seed >> 40), byte(config.Seed >> 48), byte(config.Seed >> 56),
	})

	return rand.New(source).Float32()
}

func downloadInt32ScalarForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
) int32 {
	testingObject.Helper()

	actualDType, bytes, err := backend.Download(input)
	if err != nil {
		testingObject.Fatalf("Download failed: %v", err)
	}

	if actualDType != dtype.Int32 {
		testingObject.Fatalf("download dtype mismatch: got %s want %s", actualDType, dtype.Int32)
	}

	if len(bytes) < 4 {
		testingObject.Fatalf("int32 download too short: got %d bytes", len(bytes))
	}

	return int32(binary.LittleEndian.Uint32(bytes))
}
