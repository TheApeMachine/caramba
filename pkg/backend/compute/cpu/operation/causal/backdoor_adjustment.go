package causal

import "fmt"

/*
BackdoorAdjustment computes the causal effect of X on Y adjusting for confounders Z
using the backdoor adjustment formula: P(Y|do(X)) = Σ_Z P(Y|X,Z) * P(Z).

Implemented via outcome regression: fit Y ~ X + Z via OLS, return the coefficient of X.

shape = [N_y, N_x, N_z, T]
data[0] = Y [T*N_y]
data[1] = X [T*N_x]
data[2] = Z [T*N_z]
Returns causal_effect [N_y] — mean causal effect of a unit change in X on each Y dimension.
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
func (backdoorAdjustment *BackdoorAdjustment) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 4 {
		panic(fmt.Errorf("causal: BackdoorAdjustment.Forward: len(shape)=%d, need >= 4", len(shape)).Error())
	}

	if len(data) < 3 {
		panic(fmt.Errorf("causal: BackdoorAdjustment.Forward: len(data)=%d, need >= 3", len(data)).Error())
	}

	ny, nx, nz, t := shape[0], shape[1], shape[2], shape[3]

	if len(data[0]) != t*ny {
		panic(fmt.Errorf(
			"causal: BackdoorAdjustment.Forward: len(data[0])=%d, need T*N_y=%d",
			len(data[0]), t*ny,
		).Error())
	}

	if len(data[1]) != t*nx {
		panic(fmt.Errorf(
			"causal: BackdoorAdjustment.Forward: len(data[1])=%d, need T*N_x=%d",
			len(data[1]), t*nx,
		).Error())
	}

	if len(data[2]) != t*nz {
		panic(fmt.Errorf(
			"causal: BackdoorAdjustment.Forward: len(data[2])=%d, need T*N_z=%d",
			len(data[2]), t*nz,
		).Error())
	}

	yMat := data[0]
	xMat := data[1]
	zMat := data[2]

	// Build design matrix W = [X, Z] of shape [T x (nx+nz)].
	p := nx + nz
	design := make([]float64, t*p)

	for row := 0; row < t; row++ {
		for col := 0; col < nx; col++ {
			design[row*p+col] = xMat[row*nx+col]
		}

		for col := 0; col < nz; col++ {
			design[row*p+nx+col] = zMat[row*nz+col]
		}
	}

	// OLS: beta = (W^T W)^{-1} W^T Y  shape [p x ny]
	// W^T W [p x p]
	wtw := make([]float64, p*p)
	applyMatMulTransposeLeft(wtw, design, design, t, p, p)

	wtwInv := invertSymPD(wtw, p)

	causalEffect := make([]float64, ny)

	// For each y dimension, solve for beta and sum the X coefficients.
	for yDim := 0; yDim < ny; yDim++ {
		// Extract y column [T].
		yCol := make([]float64, t)

		for row := 0; row < t; row++ {
			yCol[row] = yMat[row*ny+yDim]
		}

		// W^T y [p]
		wty := make([]float64, p)
		applyMatVecTranspose(wty, design, yCol, t, p)

		// beta = wtwInv @ wty  [p]
		beta := make([]float64, p)
		applyMatVec(beta, wtwInv, wty, p, p)

		// Causal effect of X on Y: sum of absolute X coefficients (mean over nx dims).
		effect := 0.0

		for xDim := 0; xDim < nx; xDim++ {
			effect += beta[xDim]
		}

		causalEffect[yDim] = effect / float64(nx)
	}

	return causalEffect
}
