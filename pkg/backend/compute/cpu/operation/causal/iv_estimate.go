package causal

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
IVEstimate computes the instrumental variable (2SLS) estimator for causal effects
in the presence of unmeasured confounding.

Given instrument Z (Z→X→Y, Z has no direct effect on Y):

	Stage 1: X_hat = Z @ (Z^T Z)^{-1} Z^T X
	Stage 2: beta_iv = (X_hat^T X_hat)^{-1} X_hat^T Y

shape = [T, N_z, N_x, N_y]
data[0] = Z [T*N_z] — instrument matrix
data[1] = X [T*N_x] — endogenous treatment matrix
data[2] = Y [T*N_y] — outcome matrix
Returns beta_iv [N_x*N_y] — instrumental variable estimates, row-major.
*/
type IVEstimate struct{}

/*
NewIVEstimate instantiates an IVEstimate operation.
It implements two-stage least squares via the instrumental variable approach.
*/
func NewIVEstimate() *IVEstimate {
	return &IVEstimate{}
}

/*
Forward computes the 2SLS IV estimate.
*/
func (ivEstimate *IVEstimate) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 4 {
		return nil, fmt.Errorf("causal.iv_estimate: len(shape)=%d, need >= 4", len(shape))
	}

	if err := stateDict.RequireOperationInputs("causal.iv_estimate", 3); err != nil {
		return nil, err
	}

	samples := shape[0]
	instrumentDimensions := shape[1]
	treatmentDimensions := shape[2]
	outcomeDimensions := shape[3]

	if samples <= 0 || instrumentDimensions <= 0 ||
		treatmentDimensions <= 0 || outcomeDimensions <= 0 {
		return nil, fmt.Errorf(
			"causal.iv_estimate: dimensions must be positive (T=%d N_z=%d N_x=%d N_y=%d)",
			samples, instrumentDimensions, treatmentDimensions, outcomeDimensions,
		)
	}

	if samples < instrumentDimensions {
		return nil, fmt.Errorf(
			"causal.iv_estimate: T=%d must be >= N_z=%d",
			samples, instrumentDimensions,
		)
	}

	if samples < treatmentDimensions {
		return nil, fmt.Errorf(
			"causal.iv_estimate: T=%d must be >= N_x=%d",
			samples, treatmentDimensions,
		)
	}

	zMat := stateDict.Inputs[0]
	xMat := stateDict.Inputs[1]
	yMat := stateDict.Inputs[2]

	if len(zMat) != samples*instrumentDimensions {
		return nil, fmt.Errorf(
			"causal.iv_estimate: len(Z)=%d, need T*N_z=%d",
			len(zMat), samples*instrumentDimensions,
		)
	}

	if len(xMat) != samples*treatmentDimensions {
		return nil, fmt.Errorf(
			"causal.iv_estimate: len(X)=%d, need T*N_x=%d",
			len(xMat), samples*treatmentDimensions,
		)
	}

	if len(yMat) != samples*outcomeDimensions {
		return nil, fmt.Errorf(
			"causal.iv_estimate: len(Y)=%d, need T*N_y=%d",
			len(yMat), samples*outcomeDimensions,
		)
	}

	// Stage 1: X_hat = Z (Z^T Z)^{-1} Z^T X
	// Z^T Z  [nz x nz]
	ztZ := make([]float64, instrumentDimensions*instrumentDimensions)
	applyMatMulTransposeLeft(
		ztZ,
		zMat,
		zMat,
		samples,
		instrumentDimensions,
		instrumentDimensions,
	)

	// (Z^T Z)^{-1}  [nz x nz]
	ztZInv := invertSymPD(ztZ, instrumentDimensions)

	// Z^T X  [nz x nx]
	ztX := make([]float64, instrumentDimensions*treatmentDimensions)
	applyMatMulTransposeLeft(
		ztX,
		zMat,
		xMat,
		samples,
		instrumentDimensions,
		treatmentDimensions,
	)

	// (Z^T Z)^{-1} Z^T X  [nz x nx]
	proj := make([]float64, instrumentDimensions*treatmentDimensions)
	applyMatMulFull(
		proj,
		ztZInv,
		ztX,
		instrumentDimensions,
		instrumentDimensions,
		treatmentDimensions,
	)

	// X_hat = Z @ proj  [T x nx]
	xHat := make([]float64, samples*treatmentDimensions)
	applyMatMulFull(
		xHat,
		zMat,
		proj,
		samples,
		instrumentDimensions,
		treatmentDimensions,
	)

	// Stage 2: beta_iv = (X_hat^T X_hat)^{-1} X_hat^T Y
	// X_hat^T X_hat  [nx x nx]
	xhTxh := make([]float64, treatmentDimensions*treatmentDimensions)
	applyMatMulTransposeLeft(
		xhTxh,
		xHat,
		xHat,
		samples,
		treatmentDimensions,
		treatmentDimensions,
	)

	xhTxhInv := invertSymPD(xhTxh, treatmentDimensions)

	// X_hat^T Y  [nx x ny]
	xhTy := make([]float64, treatmentDimensions*outcomeDimensions)
	applyMatMulTransposeLeft(
		xhTy,
		xHat,
		yMat,
		samples,
		treatmentDimensions,
		outcomeDimensions,
	)

	// beta_iv = (X_hat^T X_hat)^{-1} X_hat^T Y  [nx x ny]
	betaIV := make([]float64, treatmentDimensions*outcomeDimensions)
	applyMatMulFull(
		betaIV,
		xhTxhInv,
		xhTy,
		treatmentDimensions,
		treatmentDimensions,
		outcomeDimensions,
	)

	stateDict.SetOperationOutput(betaIV)

	return stateDict, nil
}
