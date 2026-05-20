//go:build ignore

package main

import (
	"fmt"
	"math"

	cpumath "github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
)

func geluFloat32Reference(value float32) float32 {
	const sqrtTwoOverTwo float32 = 0.7071067811865476
	const abramowitzP float32 = 0.3275911
	const abramowitzA1 float32 = 0.254829592
	const abramowitzA2 float32 = -0.284496736
	const abramowitzA3 float32 = 1.421413741
	const abramowitzA4 float32 = -1.453152027
	const abramowitzA5 float32 = 1.061405429
	argument := value * sqrtTwoOverTwo
	sign := float32(1)
	if argument < 0 {
		sign = -1
		argument = -argument
	}
	denominator := float32(1) + abramowitzP*argument
	numerator := argument * (((((abramowitzA5*denominator+abramowitzA4)*denominator+abramowitzA3)*denominator+abramowitzA2)*denominator + abramowitzA1) * denominator)
	exponential := float32(math.Exp(float64(-argument * argument)))
	residual := exponential * numerator
	var onePlusErf float32
	if sign < 0 {
		onePlusErf = residual
	} else {
		onePlusErf = 2 - residual
	}
	return value * 0.5 * onePlusErf
}

func ulpDistance(left, right float32) int {
	if left == right {
		return 0
	}
	if math.IsNaN(float64(left)) || math.IsNaN(float64(right)) {
		if math.IsNaN(float64(left)) && math.IsNaN(float64(right)) {
			return 0
		}
		return 1 << 30
	}
	bitsLeft := math.Float32bits(left)
	bitsRight := math.Float32bits(right)
	if bitsLeft > bitsRight {
		bitsLeft, bitsRight = bitsRight, bitsLeft
	}
	return int(bitsRight - bitsLeft)
}

func main() {
	maxULP := 0
	worst := float32(0)
	for index := 0; index < 8192; index++ {
		value := float32(1+index%240)/12 - 4
		reference := geluFloat32Reference(value)
		scalar := cpumath.FastGelu32(value)
		distance := ulpDistance(reference, scalar)
		if distance > maxULP {
			maxULP = distance
			worst = value
		}
	}
	fmt.Printf("parity-input max ULP vs FastGelu32: %d at x=%g\n", maxULP, worst)

	maxULP = 0
	for bits := uint32(0); bits < 500000; bits++ {
		value := math.Float32frombits(bits)
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			continue
		}
		reference := geluFloat32Reference(value)
		scalar := cpumath.FastGelu32(value)
		distance := ulpDistance(reference, scalar)
		if distance > maxULP {
			maxULP = distance
			worst = value
		}
	}
	fmt.Printf("float32-bit sample max ULP vs FastGelu32: %d at x=%g\n", maxULP, worst)

	for _, idx := range []int{0, 47, 100, 200, 1000} {
		value := float32(1+idx%240)/12 - 4
		reference := geluFloat32Reference(value)
		scalar := cpumath.FastGelu32(value)
		fmt.Printf("parity x=%g ref=%g scalar=%g ulp=%d\n", value, reference, scalar, ulpDistance(reference, scalar))
	}
}
