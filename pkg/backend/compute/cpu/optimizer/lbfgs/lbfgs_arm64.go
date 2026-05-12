//go:build arm64

package lbfgs

//go:noescape
func lbfgsSubNEON(dst, a, b []float64)

//go:noescape
func lbfgsDotNEON(a, b []float64) float64

//go:noescape
func lbfgsAddScaledNEON(dst, src []float64, scale float64)

//go:noescape
func lbfgsScaleNEON(dst []float64, scale float64)

//go:noescape
func lbfgsParamStepNEON(out, params, dir []float64, lr float64)

func lbfgsSub(dst, a, b []float64) {
	lbfgsSubNEON(dst, a, b)
}

func lbfgsDot(a, b []float64) float64 {
	return lbfgsDotNEON(a, b)
}

func lbfgsAddScaled(dst, src []float64, scale float64) {
	lbfgsAddScaledNEON(dst, src, scale)
}

func lbfgsScale(dst []float64, scale float64) {
	lbfgsScaleNEON(dst, scale)
}

func lbfgsParamStep(out, params, dir []float64, lr float64) {
	lbfgsParamStepNEON(out, params, dir, lr)
}
