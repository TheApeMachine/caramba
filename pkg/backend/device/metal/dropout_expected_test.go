package metal

import (
	"github.com/theapemachine/manifesto/dtype"
)

type dropoutFixture struct {
	inputBytes      []byte
	expectedBytes   []byte
	expectedFloat32 []float32
}

func dropoutFixtureForTest(elementCount int, storageDType dtype.DType) dropoutFixture {
	inputValues := dropoutInputValues(elementCount)
	inputBytes := encodeLossValuesAsDType(inputValues, storageDType)
	storedInput := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	expected := dropoutExpected(storedInput)

	return dropoutFixture{
		inputBytes:      inputBytes,
		expectedBytes:   encodeLossValuesAsDType(expected, storageDType),
		expectedFloat32: expected,
	}
}

func dropoutInputValues(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*17+11, 67, 19)
	}

	return values
}

func dropoutExpected(input []float32) []float32 {
	keepProb := float32(0.9)
	scale := float32(1.0 / keepProb)
	threshold := uint32(float64(keepProb) * (1 << 32))
	seed := metalDropoutSeedStateForTest(0xc0ffee)
	out := make([]float32, len(input))
	blockCount := len(input) &^ 3

	for index, value := range input {
		randomValue := metalDropoutRandomForIndex(index, blockCount, seed)
		if randomValue >= threshold {
			continue
		}

		out[index] = value * scale
	}

	return out
}

func metalDropoutSeedStateForTest(seed uint64) [4]uint32 {
	return [4]uint32{
		uint32(seed),
		uint32(seed >> 32),
		uint32(seed ^ 0x9e3779b9),
		uint32((seed >> 32) ^ 0x6c078965),
	}
}

func metalDropoutRandomForIndex(index int, blockCount int, seed [4]uint32) uint32 {
	if index < blockCount {
		lane := index & 3
		step := index/4 + 1

		return metalDropoutAdvanceForTest(seed[lane], step)
	}

	step := blockCount/4 + (index - blockCount) + 1
	return metalDropoutAdvanceForTest(seed[0], step)
}

func metalDropoutAdvanceForTest(seed uint32, steps int) uint32 {
	value := seed

	for range steps {
		value = metalDropoutXorshiftForTest(value)
	}

	return value
}

func metalDropoutXorshiftForTest(value uint32) uint32 {
	value ^= value << 13
	value ^= value >> 17
	value ^= value << 5

	return value
}
