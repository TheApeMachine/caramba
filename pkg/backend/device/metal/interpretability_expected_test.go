package metal

import (
	cpuinterpretability "github.com/theapemachine/caramba/pkg/backend/device/cpu/interpretability"
	"github.com/theapemachine/caramba/pkg/dtype"
)

type activationSteerFixture struct {
	baseBytes        []byte
	directionBytes   []byte
	coefficientBytes []byte
	expectedBytes    []byte
}

func activationSteerFixtureForTest(elementCount int) activationSteerFixture {
	base := activationSteerValuesForTest(elementCount, 3)
	direction := activationSteerValuesForTest(elementCount, 11)
	coefficient := float32(0.375)

	destination := make([]float32, elementCount)
	cpuinterpretability.ActivationSteerFloat32Scalar(destination, base, direction, coefficient)

	return activationSteerFixture{
		baseBytes:        encodeLossValuesAsDType(base, dtype.Float32),
		directionBytes:   encodeLossValuesAsDType(direction, dtype.Float32),
		coefficientBytes: encodeLossValuesAsDType([]float32{coefficient}, dtype.Float32),
		expectedBytes:    encodeLossValuesAsDType(destination, dtype.Float32),
	}
}

func activationSteerValuesForTest(elementCount int, salt int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = centeredPowerOfTwoValue(index*13+salt, 97, 23)
	}

	return values
}
