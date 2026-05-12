package causal

import (
	"fmt"
	"math"
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
func (backdoorAdjustment *BackdoorAdjustment) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 4 {
		panic(fmt.Errorf("causal: BackdoorAdjustment.Forward: len(shape)=%d, need >= 4", len(shape)).Error())
	}

	if len(data) < 3 {
		panic(fmt.Errorf("causal: BackdoorAdjustment.Forward: len(data)=%d, need >= 3", len(data)).Error())
	}

	ny, nx, nz, t := shape[0], shape[1], shape[2], shape[3]

	if nx <= 0 || t <= 0 {
		panic(fmt.Errorf(
			"causal: BackdoorAdjustment.Forward: need N_x > 0 and T > 0 (got N_x=%d, T=%d)",
			nx, t,
		).Error())
	}

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

	// Design W = [1, X, Z] row-major [T x p], p = 1 + nx + nz.
	p := 1 + nx + nz
	design := make([]float64, t*p)

	for row := 0; row < t; row++ {
		design[row*p] = 1.0

		for col := 0; col < nx; col++ {
			design[row*p+1+col] = xMat[row*nx+col]
		}

		for col := 0; col < nz; col++ {
			design[row*p+1+nx+col] = zMat[row*nz+col]
		}
	}

	const ridge = 1e-10

	wtw := make([]float64, p*p)
	applyMatMulTransposeLeft(wtw, design, design, t, p, p)
	addRidgeToDiagInPlace(wtw, p, ridge)

	wtwInv := invertSymPD(wtw, p)

	causalEffect := make([]float64, ny)

	yCol := make([]float64, t)
	wty := make([]float64, p)
	beta := make([]float64, p)

	for yDim := 0; yDim < ny; yDim++ {
		for row := range yCol {
			yCol[row] = yMat[row*ny+yDim]
		}

		for pIdx := range wty {
			wty[pIdx] = 0
		}

		applyMatVecTranspose(wty, design, yCol, t, p)

		for pIdx := range beta {
			beta[pIdx] = 0
		}

		applyMatVec(beta, wtwInv, wty, p, p)

		effect := 0.0

		for xDim := 0; xDim < nx; xDim++ {
			effect += math.Abs(beta[1+xDim])
		}

		causalEffect[yDim] = effect / float64(nx)
	}

	return causalEffect
}
