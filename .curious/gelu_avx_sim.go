//go:build ignore

package main

import (
	"fmt"
	"math"

	cpumath "github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func geluAVX512Path(value float32) float32 {
	const sqrt2inv float32 = 0.7071067811865475
	const eps float32 = 0.01
	const one float32 = 1.0
	const half float32 = 0.5

	saved := value
	t := value * sqrt2inv
	absT := float32(math.Abs(float64(t)))
	ratio := (absT + eps) / absT

	inner := t*ratio + t
	inner *= ratio
	v := one - inner

	var factor float32
	if t*t <= t {
		factor = one + v
	} else {
		factor = one
	}

	return saved * factor * half
}

func main() {
	maxULP := 0
	for index := 0; index < 8192; index++ {
		value := float32(1+index%240)/12 - 4
		ref := cpumath.FastGelu32(value)
		got := geluAVX512Path(value)
		ulp := parity.Float32ULPDistance(ref, got)
		if ulp > maxULP {
			maxULP = ulp
			if ulp > 2 {
				fmt.Printf("bad index=%d x=%g ref=%g got=%g ulp=%d\n", index, value, ref, got, ulp)
			}
		}
	}
	fmt.Printf("AVX512-path max ULP=%d\n", maxULP)
}
