//go:build ignore

package main

import (
	"fmt"
	"math"

	cpumath "github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func geluFloat32Reference(value float32) float32 {
	const sqrtTwoOverTwo = float32(0.7071067811865475)
	scaled := value * sqrtTwoOverTwo
	axAbs := float32(math.Abs(float64(scaled)))
	reciprocal := float32(1) / (1 + 0.3275911*axAbs)
	polynomial := (((((1.061405429*reciprocal - 1.453152027) * reciprocal) +
		1.421413741) * reciprocal - 0.284496736) * reciprocal + 0.254829592) * reciprocal
	exponential := float32(math.Exp(float64(-0.5 * axAbs * axAbs)))
	exponential = exponential * exponential
	residual := polynomial * exponential
	var onePlusErf float32
	if scaled < 0 {
		onePlusErf = residual
	} else {
		onePlusErf = 2 - residual
	}
	return 0.5 * value * onePlusErf
}

func main() {
	maxULP := 0
	worstIndex := 0
	for index := 0; index < 8192; index++ {
		value := float32(1+index%240)/12 - 4
		reference := geluFloat32Reference(value)
		scalar := cpumath.FastGelu32(value)
		distance := parity.Float32ULPDistance(reference, scalar)
		if distance > maxULP {
			maxULP = distance
			worstIndex = index
		}
	}
	value := float32(1+worstIndex%240)/12 - 4
	fmt.Printf("max ULP ref vs FastGelu32 on parity inputs: %d at index %d x=%g ref=%g scalar=%g\n",
		maxULP, worstIndex, value, geluFloat32Reference(value), cpumath.FastGelu32(value))

	maxRel := float64(0)
	for index := 0; index < 8192; index++ {
		value := float32(1+index%240)/12 - 4
		reference := geluFloat32Reference(value)
		scalar := cpumath.FastGelu32(value)
		rel := math.Abs(float64(reference - scalar))
		if rel > maxRel {
			maxRel = rel
		}
	}
	fmt.Printf("max abs diff on parity inputs: %g (CPU GeluF32Generic bar is 2 ULP)\n", maxRel)

	for _, index := range []int{47, 95, 143} {
		value := float32(1+index%240)/12 - 4
		reference := geluFloat32Reference(value)
		scalar := cpumath.FastGelu32(value)
		fmt.Printf("index %d x=%g ref=%g scalar=%g ulp=%d\n",
			index, value, reference, scalar, parity.Float32ULPDistance(reference, scalar))
	}
}
