package rmsprop

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
RMSProp — running average of squared gradients. All four variants (plain,
centered, momentum, centered-momentum) execute through dedicated AVX2/SSE2/NEON
kernels with the entire update pipeline fused.
*/
type RMSProp struct {
}

func NewRMSProp() *RMSProp {
	return &RMSProp{}
}

func (rms *RMSProp) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("rmsprop"); err != nil {
		return nil, err
	}

	stateDict.EnsureBuf()

	if stateDict.Centered {
		stateDict.EnsureGradAvg()
	}

	switch {
	case stateDict.Centered && stateDict.Momentum != 0:
		rmspropCenteredMomentum(
			stateDict.Out,
			stateDict.V,
			stateDict.GradAvg,
			stateDict.Buf,
			stateDict.Params,
			stateDict.Grads,
			stateDict.LR,
			stateDict.Alpha,
			stateDict.Eps,
			stateDict.Momentum,
			stateDict.WD,
		)
	case stateDict.Centered:
		rmspropCentered(
			stateDict.Out,
			stateDict.V,
			stateDict.GradAvg,
			stateDict.Params,
			stateDict.Grads,
			stateDict.LR,
			stateDict.Alpha,
			stateDict.Eps,
			stateDict.WD,
		)
	case stateDict.Momentum != 0:
		rmspropMomentum(
			stateDict.Out,
			stateDict.V,
			stateDict.Buf,
			stateDict.Params,
			stateDict.Grads,
			stateDict.LR,
			stateDict.Alpha,
			stateDict.Eps,
			stateDict.Momentum,
			stateDict.WD,
		)
	default:
		rmspropPlain(
			stateDict.Out,
			stateDict.V,
			stateDict.Params,
			stateDict.Grads,
			stateDict.LR,
			stateDict.Alpha,
			stateDict.Eps,
			stateDict.WD,
		)
	}

	return stateDict, nil
}
