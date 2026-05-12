package activation

import (
	"fmt"
)

/*
Tail handlers for non-aligned remainders (1–3 elements per SIMD batch). Every
function dispatches to a dedicated assembly kernel that fuses the polynomial
exp/tanh/sigmoid math inline — no Go-side scalar transcendentals.
*/

// Per-arch kernel implementations live in scalar_amd64.go / scalar_arm64.go.

func scalarTanh(dst, src []float64) {
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}

	scalarTanhKernel(dst[:n], src[:n])
}

func scalarSigmoid(dst, src []float64) {
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}

	scalarSigmoidKernel(dst[:n], src[:n])
}

func scalarSigmoidAt(value float64) float64 {
	in := [1]float64{value}
	out := [1]float64{}
	scalarSigmoidKernel(out[:], in[:])

	return out[0]
}

func scalarReLU(dst, src []float64) {
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}

	scalarReLUKernel(dst[:n], src[:n])
}

func scalarLeakyReLU(dst, src []float64, alpha float64) {
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}

	scalarLeakyReLUKernel(dst[:n], src[:n], alpha)
}

func scalarGeLU(dst, src []float64) {
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}

	scalarGeLUKernel(dst[:n], src[:n])
}

func scalarSwiGLU(dst, src []float64) {
	half := len(dst)

	if len(src) < 2*half {
		panic(fmt.Sprintf(
			"scalarSwiGLU: src too short for gates|values layout (len(dst)=%d len(src)=%d, need %d)",
			half, len(src), 2*half,
		))
	}

	scalarSwiGLUKernel(dst, src[:2*half])
}
