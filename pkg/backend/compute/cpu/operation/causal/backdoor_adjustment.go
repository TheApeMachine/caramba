package causal

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
BackdoorAdjustment computes the causal effect of X on Y adjusting for confounders Z
using the backdoor adjustment formula: P(Y|do(X)) = Σ_Z P(Y|X,Z) * P(Z).

Implemented via outcome regression: fit Y ~ 1 + X + Z via OLS; aggregate absolute X coefficients.

shape = [N_y, N_x, N_z, T]
data[0] = Y [T*N_y]
data[1] = X [T*N_x]
data[2] = Z [T*N_z]
Returns causal_effect [N_y] — mean magnitude of causal X effect per Y dimension.
*/
type BackdoorAdjustment struct{}

/*
NewBackdoorAdjustment instantiates a BackdoorAdjustment operation.
It implements the backdoor criterion for estimating causal effects.
*/
func NewBackdoorAdjustment() *BackdoorAdjustment {
	return &BackdoorAdjustment{}
}

/*
Forward computes the backdoor-adjusted causal effect.
*/
func (backdoorAdjustment *BackdoorAdjustment) Forward(
	stateDict *state.Dict,
) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 4 {
		return nil, fmt.Errorf("causal.backdoor_adjustment: len(shape)=%d, need >= 4", len(shape))
	}

	if err := stateDict.RequireOperationInputs("causal.backdoor_adjustment", 3); err != nil {
		return nil, err
	}

	outcomeDimensions := shape[0]
	treatmentDimensions := shape[1]
	conFounderDimensions := shape[2]
	samples := shape[3]

	if outcomeDimensions <= 0 || treatmentDimensions <= 0 || conFounderDimensions < 0 || samples <= 0 {
		return nil, fmt.Errorf(
			"causal.backdoor_adjustment: need N_y > 0, N_x > 0, N_z >= 0, T > 0 (got N_y=%d N_x=%d N_z=%d T=%d)",
			outcomeDimensions, treatmentDimensions, conFounderDimensions, samples,
		)
	}

	if len(stateDict.Inputs[0]) != samples*outcomeDimensions {
		return nil, fmt.Errorf(
			"causal.backdoor_adjustment: len(data[0])=%d, need T*N_y=%d",
			len(stateDict.Inputs[0]), samples*outcomeDimensions,
		)
	}

	if len(stateDict.Inputs[1]) != samples*treatmentDimensions {
		return nil, fmt.Errorf(
			"causal.backdoor_adjustment: len(data[1])=%d, need T*N_x=%d",
			len(stateDict.Inputs[1]), samples*treatmentDimensions,
		)
	}

	if len(stateDict.Inputs[2]) != samples*conFounderDimensions {
		return nil, fmt.Errorf(
			"causal.backdoor_adjustment: len(data[2])=%d, need T*N_z=%d",
			len(stateDict.Inputs[2]), samples*conFounderDimensions,
		)
	}

	yMat := stateDict.Inputs[0]
	xMat := stateDict.Inputs[1]
	zMat := stateDict.Inputs[2]

	// Design W = [1, X, Z] row-major [T x p], p = 1 + nx + nz.
	p := 1 + treatmentDimensions + conFounderDimensions
	design := make([]float64, samples*p)

	for row := 0; row < samples; row++ {
		design[row*p] = 1.0

		for col := 0; col < treatmentDimensions; col++ {
			design[row*p+1+col] = xMat[row*treatmentDimensions+col]
		}

		for col := 0; col < conFounderDimensions; col++ {
			design[row*p+1+treatmentDimensions+col] = zMat[row*conFounderDimensions+col]
		}
	}

	const ridge = 1e-10

	wtw := make([]float64, p*p)
	applyMatMulTransposeLeft(wtw, design, design, samples, p, p)
	addRidgeToDiagInPlace(wtw, p, ridge)

	wtwInv := invertSymPD(wtw, p)

	causalEffect := make([]float64, outcomeDimensions)

	yCol := make([]float64, samples)
	wty := make([]float64, p)
	beta := make([]float64, p)

	for yDim := 0; yDim < outcomeDimensions; yDim++ {
		for row := range yCol {
			yCol[row] = yMat[row*outcomeDimensions+yDim]
		}

		for pIdx := range wty {
			wty[pIdx] = 0
		}

		applyMatVecTranspose(wty, design, yCol, samples, p)

		for pIdx := range beta {
			beta[pIdx] = 0
		}

		applyMatVec(beta, wtwInv, wty, p, p)

		effect := 0.0

		for xDim := 0; xDim < treatmentDimensions; xDim++ {
			effect += math.Abs(beta[1+xDim])
		}

		causalEffect[yDim] = effect / float64(treatmentDimensions)
	}

	stateDict.SetOperationOutput(causalEffect)

	return stateDict, nil
}
