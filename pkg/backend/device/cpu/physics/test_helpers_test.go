package physics

import (
	"math"
	"math/rand"
)

func randomPhysicsFloat32(length int, seed int64) []float32 {
	randomSource := rand.New(rand.NewSource(seed))
	values := make([]float32, length)

	for index := range values {
		values[index] = float32(randomSource.NormFloat64()) * 0.25
	}

	return values
}

func randomPhysicsDensity(length int, seed int64) []float32 {
	randomSource := rand.New(rand.NewSource(seed))
	values := make([]float32, length)

	for index := range values {
		value := float32(math.Abs(randomSource.NormFloat64())) + 0.05
		values[index] = value
	}

	return values
}

func physicsInvH2ForTest() float32 {
	return 4.0
}

func physicsInvTwoDxForTest() float32 {
	return 0.5
}

func physicsInvDenForTest() float32 {
	return 0.125
}

func physicsQuantumScaleForTest() float32 {
	return float32(-0.5)
}
