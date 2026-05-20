//go:build ignore

package main

import (
	"fmt"

	cpumath "github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/activation"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func main() {
	maxULP := 0
	worstIndex := 0
	var worstX float32

	for index := 0; index < 8192; index++ {
		value := float32(1+index%240)/12 - 4
		source := []float32{value}
		want := make([]float32, 1)
		got := make([]float32, 1)
		activation.GeluF32Generic(&want[0], &source[0], 1)
		activation.GeluF32NEON(&got[0], &source[0], 1)
		ulp := parity.Float32ULPDistance(want[0], got[0])
		if ulp > maxULP {
			maxULP = ulp
			worstIndex = index
			worstX = value
		}
	}

	value := float32(1+worstIndex%240)/12 - 4
	source := []float32{value}
	want := make([]float32, 1)
	got := make([]float32, 1)
	activation.GeluF32Generic(&want[0], &source[0], 1)
	activation.GeluF32NEON(&got[0], &source[0], 1)

	fmt.Printf("NEON vs Generic max ULP=%d at index=%d x=%g\n", maxULP, worstIndex, worstX)
	fmt.Printf("generic=%g neon=%g fast=%g\n", want[0], got[0], cpumath.FastGelu32(value))
}
