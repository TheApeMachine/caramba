//go:build amd64

package lbfgs

import "golang.org/x/sys/cpu"

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func lbfgsSubAVX2(dst, a, b []float64)

//go:noescape
func lbfgsSubSSE2(dst, a, b []float64)

//go:noescape
func lbfgsDotAVX2(a, b []float64) float64

//go:noescape
func lbfgsDotSSE2(a, b []float64) float64

//go:noescape
func lbfgsAddScaledAVX2(dst, src []float64, scale float64)

//go:noescape
func lbfgsAddScaledSSE2(dst, src []float64, scale float64)

//go:noescape
func lbfgsScaleAVX2(dst []float64, scale float64)

//go:noescape
func lbfgsScaleSSE2(dst []float64, scale float64)

//go:noescape
func lbfgsParamStepAVX2(out, params, dir []float64, lr float64)

//go:noescape
func lbfgsParamStepSSE2(out, params, dir []float64, lr float64)

func lbfgsSub(dst, a, b []float64) {
	if useAVX2 && useFMA {
		lbfgsSubAVX2(dst, a, b)
		return
	}

	lbfgsSubSSE2(dst, a, b)
}

func lbfgsDot(a, b []float64) float64 {
	if useAVX2 && useFMA {
		return lbfgsDotAVX2(a, b)
	}

	return lbfgsDotSSE2(a, b)
}

func lbfgsAddScaled(dst, src []float64, scale float64) {
	if useAVX2 && useFMA {
		lbfgsAddScaledAVX2(dst, src, scale)
		return
	}

	lbfgsAddScaledSSE2(dst, src, scale)
}

func lbfgsScale(dst []float64, scale float64) {
	if useAVX2 {
		lbfgsScaleAVX2(dst, scale)
		return
	}

	lbfgsScaleSSE2(dst, scale)
}

func lbfgsParamStep(out, params, dir []float64, lr float64) {
	if useAVX2 && useFMA {
		lbfgsParamStepAVX2(out, params, dir, lr)
		return
	}

	lbfgsParamStepSSE2(out, params, dir, lr)
}
