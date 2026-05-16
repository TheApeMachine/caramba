package hebbian

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Hebbian / Oja / BCM rules. Each Step delegates to a fused AVX2/SSE2/NEON
kernel; weight update, norm computation, and max-norm rescale all stay in
architecture-specific execution.
*/
type Hebbian struct {
}

func NewHebbian() *Hebbian {
	return &Hebbian{}
}

func (hebb *Hebbian) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("hebbian"); err != nil {
		return nil, err
	}

	if stateDict.MaxNorm > 0 {
		norm := hebbStepNorm(stateDict.Out, stateDict.Params, stateDict.Grads, stateDict.LR)

		if norm > stateDict.MaxNorm {
			hebbScale(stateDict.Out, stateDict.MaxNorm/norm)
		}

		return stateDict, nil
	}

	hebbStep(stateDict.Out, stateDict.Params, stateDict.Grads, stateDict.LR)

	return stateDict, nil
}

/*
OjaRule extends Hebb with a decay term that keeps weights on the unit sphere.

	ΔW = η * (post*pre - post²*W)
*/
type OjaRule struct {
	LR float64
}

func NewOjaRule(lr float64) *OjaRule {
	return &OjaRule{LR: lr}
}

func (oja *OjaRule) Step(params, grads []float64) []float64 {
	postSq := reduceSumSq(grads)
	out := make([]float64, len(params))
	ojaStep(out, params, grads, oja.LR, postSq)

	return out
}

/*
BCM (Bienenstock-Cooper-Munro) — sliding threshold.

	θ  = τ⁻¹ * E[post²]
	ΔW = η * post * (post - θ) * pre
*/
type BCM struct {
	LR    float64
	Tau   float64
	theta float64
}

func NewBCM(lr, tau float64) *BCM {
	return &BCM{LR: lr, Tau: tau}
}

func (bcm *BCM) Step(params, grads []float64) []float64 {
	postSq := reduceSumSq(grads)
	bcm.theta += (postSq - bcm.theta) / bcm.Tau
	factor := bcm.LR * (postSq - bcm.theta)
	out := make([]float64, len(params))
	hebbStep(out, params, grads, factor)

	return out
}
