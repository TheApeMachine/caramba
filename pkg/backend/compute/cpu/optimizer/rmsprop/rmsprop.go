package rmsprop

/*
RMSProp — running average of squared gradients. All four variants (plain,
centered, momentum, centered-momentum) execute through dedicated AVX2/SSE2/NEON
kernels with the entire update pipeline fused.
*/
type RMSProp struct {
	LR       float64
	Alpha    float64
	Eps      float64
	Momentum float64
	WD       float64
	Centered bool
	v, buf   []float64
	gradAvg  []float64
}

func NewRMSProp(lr, alpha, eps, momentum, wd float64, centered bool) *RMSProp {
	return &RMSProp{LR: lr, Alpha: alpha, Eps: eps, Momentum: momentum, WD: wd, Centered: centered}
}

func (rms *RMSProp) Step(params, grads []float64) []float64 {
	n := len(params)

	if rms.v == nil {
		rms.v = make([]float64, n)
		rms.buf = make([]float64, n)

		if rms.Centered {
			rms.gradAvg = make([]float64, n)
		}
	}

	out := make([]float64, n)

	switch {
	case rms.Centered && rms.Momentum != 0:
		rmspropCenteredMomentum(out, rms.v, rms.gradAvg, rms.buf, params, grads, rms.LR, rms.Alpha, rms.Eps, rms.Momentum, rms.WD)
	case rms.Centered:
		rmspropCentered(out, rms.v, rms.gradAvg, params, grads, rms.LR, rms.Alpha, rms.Eps, rms.WD)
	case rms.Momentum != 0:
		rmspropMomentum(out, rms.v, rms.buf, params, grads, rms.LR, rms.Alpha, rms.Eps, rms.Momentum, rms.WD)
	default:
		rmspropPlain(out, rms.v, params, grads, rms.LR, rms.Alpha, rms.Eps, rms.WD)
	}

	return out
}
