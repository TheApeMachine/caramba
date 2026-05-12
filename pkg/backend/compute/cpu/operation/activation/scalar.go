package activation

import (
	"fmt"
	"math"

	mathops "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

/*
Tail handlers for non-aligned remainders (1–3 elements per call). Each pads to
SIMD width, swaps out NaN inputs for a sentinel zero (preserving NaN positions
in a mask), runs the math via SIMD primitives (ExpVec/SignVec/MulVec/...),
then restores NaN to flagged lanes. No scalar transcendental survives on this
path; everything reaches an assembly kernel.
*/

const tailPad = 4

// stagePad copies src into a tailPad scratch, masks NaN to zero, and returns
// the scratch (sized tailPad), a nan-presence flag-per-lane slice (length n),
// and the effective element count n. Callers run SIMD on the full scratch and
// reapply NaN before copy-out via finalizePad.
func stagePad(src []float64) (scratch [tailPad]float64, nanMask [tailPad]bool, n int) {
	n = len(src)

	if n > tailPad {
		n = tailPad
	}

	for idx := 0; idx < n; idx++ {
		value := src[idx]

		if math.IsNaN(value) {
			nanMask[idx] = true
			scratch[idx] = 0
			continue
		}

		scratch[idx] = value
	}

	return
}

// finalizePad copies the first n elements of scratch into dst, replacing
// positions in nanMask with NaN.
func finalizePad(dst []float64, scratch [tailPad]float64, nanMask [tailPad]bool, n int) {
	if len(dst) < n {
		n = len(dst)
	}

	for idx := 0; idx < n; idx++ {
		if nanMask[idx] {
			dst[idx] = math.NaN()
			continue
		}

		dst[idx] = scratch[idx]
	}
}

func scalarGeLU(dst, src []float64) {
	if len(src) == 0 || len(dst) == 0 {
		return
	}

	scratch, nanMask, n := stagePad(src)

	// y = 0.5 * x * (1 + tanh(z)), z = sqrt(2/π)*(x + 0.044715 x³)
	var x3 [tailPad]float64
	mathops.MulVec(x3[:], scratch[:], scratch[:])
	mathops.MulVec(x3[:], x3[:], scratch[:])
	mathops.ScaleVec(x3[:], 0.044715)

	var z [tailPad]float64
	copy(z[:], scratch[:])
	mathops.AddScaledVec(z[:], x3[:], 1.0)
	mathops.ScaleVec(z[:], 0.7978845608028654)
	// tanh(z) = (e^{2z} - 1) / (e^{2z} + 1)
	mathops.ScaleVec(z[:], 2.0)
	mathops.ExpVec(z[:], z[:])

	var num [tailPad]float64
	copy(num[:], z[:])
	mathops.AddScalarVec(num[:], -1)
	mathops.AddScalarVec(z[:], 1)
	mathops.DivVec(num[:], num[:], z[:])
	mathops.AddScalarVec(num[:], 1)
	mathops.MulVec(scratch[:], scratch[:], num[:])
	mathops.ScaleVec(scratch[:], 0.5)

	finalizePad(dst, scratch, nanMask, n)
}

func scalarLeakyReLU(dst, src []float64, alpha float64) {
	if len(src) == 0 || len(dst) == 0 {
		return
	}

	scratch, nanMask, n := stagePad(src)

	// blend = α + (1-α)*posMask, where posMask = (sign(x)+1)/2
	var sg [tailPad]float64
	mathops.SignVec(sg[:], scratch[:])
	mathops.AddScalarVec(sg[:], 1)
	mathops.ScaleVec(sg[:], 0.5)
	mathops.ScaleVec(sg[:], 1-alpha)
	mathops.AddScalarVec(sg[:], alpha)
	mathops.MulVec(scratch[:], scratch[:], sg[:])

	finalizePad(dst, scratch, nanMask, n)
}

func scalarReLU(dst, src []float64) {
	if len(src) == 0 || len(dst) == 0 {
		return
	}

	scratch, nanMask, n := stagePad(src)

	mathops.ClampVec(scratch[:], 0, math.MaxFloat64)

	finalizePad(dst, scratch, nanMask, n)
}

func scalarSigmoid(dst, src []float64) {
	if len(src) == 0 || len(dst) == 0 {
		return
	}

	scratch, nanMask, n := stagePad(src)

	// 1/(1+exp(-x))
	var t [tailPad]float64
	copy(t[:], scratch[:])
	mathops.ScaleVec(t[:], -1)
	mathops.ExpVec(t[:], t[:])
	mathops.AddScalarVec(t[:], 1)

	var one [tailPad]float64
	for idx := range one {
		one[idx] = 1
	}

	mathops.DivVec(scratch[:], one[:], t[:])

	finalizePad(dst, scratch, nanMask, n)
}

func scalarSigmoidAt(value float64) float64 {
	in := [1]float64{value}
	out := [1]float64{}
	scalarSigmoid(out[:], in[:])

	return out[0]
}

func scalarSwiGLU(dst, src []float64) {
	half := len(dst)

	if len(src) < 2*half {
		panic(fmt.Sprintf(
			"scalarSwiGLU: src too short for gates|values layout (len(dst)=%d len(src)=%d, need %d)",
			half, len(src), 2*half,
		))
	}

	gates := src[:half]
	values := src[half : 2*half]
	gated := make([]float64, half)
	scalarSigmoid(gated, gates)
	mathops.MulVec(dst, gated, values)
}

func scalarTanh(dst, src []float64) {
	if len(src) == 0 || len(dst) == 0 {
		return
	}

	scratch, nanMask, n := stagePad(src)

	// Detect ±Inf for explicit clamp to ±1 (otherwise (e^{2*Inf}-1)/(e^{2*Inf}+1)
	// becomes (Inf-1)/(Inf+1) which IEEE renders as NaN before our NaN restore).
	var infSign [tailPad]float64

	for idx := 0; idx < n; idx++ {
		switch {
		case math.IsInf(scratch[idx], +1):
			infSign[idx] = +1
			scratch[idx] = 0
		case math.IsInf(scratch[idx], -1):
			infSign[idx] = -1
			scratch[idx] = 0
		}
	}

	// tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
	var e [tailPad]float64
	copy(e[:], scratch[:])
	mathops.ScaleVec(e[:], 2.0)
	mathops.ExpVec(e[:], e[:])

	var num [tailPad]float64
	copy(num[:], e[:])
	mathops.AddScalarVec(num[:], -1)
	mathops.AddScalarVec(e[:], 1)
	mathops.DivVec(scratch[:], num[:], e[:])

	for idx := 0; idx < n; idx++ {
		if infSign[idx] != 0 {
			scratch[idx] = infSign[idx]
		}
	}

	finalizePad(dst, scratch, nanMask, n)
}
