package adam

import (
	stdmath "math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Adam implements the Adam optimizer (Kingma & Ba, 2015).
m  = β1*m + (1-β1)*g
v  = β2*v + (1-β2)*g²
m̂  = m / (1-β1^t)
v̂  = v / (1-β2^t)
p -= lr * m̂ / (sqrt(v̂) + ε)

The hot path runs through dedicated AVX2/SSE2/NEON kernels that fuse the
moment updates, sqrt, divide, decoupled weight decay (AdamW), and parameter
step inline — no Go-side scalar arithmetic touches per-element data.
*/
type Adam struct{}

func NewAdam() *Adam {
	return &Adam{}
}

func (adam *Adam) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("adam"); err != nil {
		return nil, err
	}

	stateDict.Step++

	lrT := biasCorrectedLR(stateDict)

	adamKernel(
		stateDict.Out,
		stateDict.M,
		stateDict.V,
		stateDict.Params,
		stateDict.Grads,
		stateDict.Beta1,
		stateDict.Beta2,
		lrT,
		stateDict.Eps,
		0,
	)

	return stateDict, nil
}

func biasCorrectedLR(stateDict *state.Dict) float64 {
	bc1 := 1 - stdmath.Pow(stateDict.Beta1, float64(stateDict.Step))
	bc2 := 1 - stdmath.Pow(stateDict.Beta2, float64(stateDict.Step))
	return stateDict.LR * stdmath.Sqrt(bc2) / bc1
}
