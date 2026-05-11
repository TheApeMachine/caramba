package causal

import "fmt"

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
func (ivEstimate *IVEstimate) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 4 {
		panic(fmt.Errorf("causal: IVEstimate.Forward: len(shape)=%d, need >= 4", len(shape)).Error())
	}

	if len(data) < 3 {
		panic(fmt.Errorf("causal: IVEstimate.Forward: len(data)=%d, need >= 3", len(data)).Error())
	}

	t, nz, nx, ny := shape[0], shape[1], shape[2], shape[3]

	zMat := data[0]
	xMat := data[1]
	yMat := data[2]

	if len(zMat) != t*nz {
		panic(fmt.Errorf(
			"causal: IVEstimate.Forward: len(Z)=%d, need T*N_z=%d",
			len(zMat), t*nz,
		).Error())
	}

	if len(xMat) != t*nx {
		panic(fmt.Errorf(
			"causal: IVEstimate.Forward: len(X)=%d, need T*N_x=%d",
			len(xMat), t*nx,
		).Error())
	}

	if len(yMat) != t*ny {
		panic(fmt.Errorf(
			"causal: IVEstimate.Forward: len(Y)=%d, need T*N_y=%d",
			len(yMat), t*ny,
		).Error())
	}

	// Stage 1: X_hat = Z (Z^T Z)^{-1} Z^T X
	// Z^T Z  [nz x nz]
	ztZ := make([]float64, nz*nz)
	applyMatMulTransposeLeft(ztZ, zMat, zMat, t, nz, nz)

	// (Z^T Z)^{-1}  [nz x nz]
	ztZInv := invertSymPD(ztZ, nz)

	// Z^T X  [nz x nx]
	ztX := make([]float64, nz*nx)
	applyMatMulTransposeLeft(ztX, zMat, xMat, t, nz, nx)

	// (Z^T Z)^{-1} Z^T X  [nz x nx]
	proj := make([]float64, nz*nx)
	applyMatMulFull(proj, ztZInv, ztX, nz, nz, nx)

	// X_hat = Z @ proj  [T x nx]
	xHat := make([]float64, t*nx)
	applyMatMulFull(xHat, zMat, proj, t, nz, nx)

	// Stage 2: beta_iv = (X_hat^T X_hat)^{-1} X_hat^T Y
	// X_hat^T X_hat  [nx x nx]
	xhTxh := make([]float64, nx*nx)
	applyMatMulTransposeLeft(xhTxh, xHat, xHat, t, nx, nx)

	xhTxhInv := invertSymPD(xhTxh, nx)

	// X_hat^T Y  [nx x ny]
	xhTy := make([]float64, nx*ny)
	applyMatMulTransposeLeft(xhTy, xHat, yMat, t, nx, ny)

	// beta_iv = (X_hat^T X_hat)^{-1} X_hat^T Y  [nx x ny]
	betaIV := make([]float64, nx*ny)
	applyMatMulFull(betaIV, xhTxhInv, xhTy, nx, nx, ny)

	return betaIV
}
