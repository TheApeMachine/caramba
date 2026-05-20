//go:build ignore

package main

import (
	"fmt"
	"math"

	cpumath "github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func main() {
	x := float32(1.0/12.0 - 4.0)
	ref := cpumath.FastGelu32(x)
	got := geluSoftfloat32Full(x)
	fmt.Printf("x=%g ref=%g got=%g ulp=%d\n", x, ref, got, parity.Float32ULPDistance(ref, got))

	maxULP := 0
	for index := range 8192 {
		input := float32(1+index%240)/12 - 4
		reference := cpumath.FastGelu32(input)
		actual := geluSoftfloat32Full(input)
		distance := int(parity.Float32ULPDistance(reference, actual))
		if distance > maxULP {
			maxULP = distance
		}
		if math.IsNaN(float64(actual)) {
			fmt.Printf("NaN at index %d input %g\n", index, input)
			break
		}
	}
	fmt.Printf("maxULP=%d\n", maxULP)
}
