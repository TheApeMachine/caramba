package metal

import (
	cpumodelediting "github.com/theapemachine/caramba/pkg/backend/device/cpu/model_editing"
	"github.com/theapemachine/caramba/pkg/dtype"
)

type weightGraftFixture struct {
	weightsBytes   []byte
	injectionBytes []byte
	expectedBytes  []byte
}

func weightGraftFixtureForTest(elementCount int) weightGraftFixture {
	weights := weightGraftValuesForTest(elementCount, 5)
	injection := weightGraftValuesForTest(elementCount, 17)

	want := make([]float32, elementCount)
	copy(want, weights)
	cpumodelediting.WeightGraftAddFloat32Scalar(want, injection)

	return weightGraftFixture{
		weightsBytes:   encodeLossValuesAsDType(weights, dtype.Float32),
		injectionBytes: encodeLossValuesAsDType(injection, dtype.Float32),
		expectedBytes:  encodeLossValuesAsDType(want, dtype.Float32),
	}
}

func weightGraftValuesForTest(elementCount int, salt int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*19+salt, 101, 29)
	}

	return values
}
