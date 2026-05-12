package causal

import (
	"fmt"
	"math"
)

/*
CATE computes the Conditional Average Treatment Effect:
E[Y(1) - Y(0) | X=x] via outcome regression on treatment subgroups.

Fits linear models E[Y|T=1, X=x] and E[Y|T=0, X=x] separately with intercept,
then computes CATE(x) = mu_1(x) - mu_0(x) for each observation.

shape = [T, N_x, 1]
data[0] = X [T*N_x]      — covariate matrix
data[1] = T_treatment [T] — binary treatment indicator (0 or 1)
data[2] = Y [T]           — outcome vector
Returns cate [T] — individual-level CATE estimates (NaN if treated-only or control-only).
*/
type CATE struct{}

/*
NewCATE instantiates a CATE operation.
It estimates heterogeneous treatment effects via outcome regression.
*/
func NewCATE() *CATE {
	return &CATE{}
}

/*
Forward computes the CATE for each observation.
*/
func (cate *CATE) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Errorf("causal: CATE.Forward: len(shape)=%d, need >= 2", len(shape)).Error())
	}

	if len(data) < 3 {
		panic(fmt.Errorf("causal: CATE.Forward: len(data)=%d, need >= 3", len(data)).Error())
	}

	t := shape[0]
	nx := shape[1]

	xMat := data[0]
	treatment := data[1]
	yVec := data[2]

	if len(xMat) != t*nx {
		panic(fmt.Errorf(
			"causal: CATE.Forward: len(X)=%d, need T*N_x=%d",
			len(xMat), t*nx,
		).Error())
	}

	if len(treatment) != t || len(yVec) != t {
		panic(fmt.Errorf(
			"causal: CATE.Forward: T_treatment and Y must have len T=%d", t,
		).Error())
	}

	treatedIdx := make([]int, 0, t)
	controlIdx := make([]int, 0, t)

	for obsIdx := 0; obsIdx < t; obsIdx++ {
		if treatment[obsIdx] >= 0.5 {
			treatedIdx = append(treatedIdx, obsIdx)
		} else {
			controlIdx = append(controlIdx, obsIdx)
		}
	}

	cateValues := make([]float64, t)

	if len(treatedIdx) == 0 || len(controlIdx) == 0 {
		for obsIdx := range cateValues {
			cateValues[obsIdx] = math.NaN()
		}

		return cateValues
	}

	beta1 := fitOLS(xMat, yVec, treatedIdx, nx)
	beta0 := fitOLS(xMat, yVec, controlIdx, nx)

	for obsIdx := 0; obsIdx < t; obsIdx++ {
		xRow := xMat[obsIdx*nx : (obsIdx+1)*nx]
		mu1 := beta1[0] + dotProduct(beta1[1:], xRow)
		mu0 := beta0[0] + dotProduct(beta0[1:], xRow)
		cateValues[obsIdx] = mu1 - mu0
	}

	return cateValues
}

/*
fitOLS fits Y ~ 1 + X on the subset of observations given by indices.
Returns beta [1+nx]: beta[0] intercept, beta[1:] covariate slopes.
*/
func fitOLS(xMat, yVec []float64, indices []int, nx int) []float64 {
	nFeat := nx + 1
	n := len(indices)

	if n == 0 {
		coef := make([]float64, nFeat)

		for idx := range coef {
			coef[idx] = math.NaN()
		}

		return coef
	}

	xSub := make([]float64, n*nFeat)
	ySub := make([]float64, n)

	for subIdx, obsIdx := range indices {
		rowOff := subIdx * nFeat
		xSub[rowOff] = 1.0
		copy(xSub[rowOff+1:(subIdx+1)*nFeat], xMat[obsIdx*nx:(obsIdx+1)*nx])
		ySub[subIdx] = yVec[obsIdx]
	}

	const ridge = 1e-10

	xtx := make([]float64, nFeat*nFeat)
	applyMatMulTransposeLeft(xtx, xSub, xSub, n, nFeat, nFeat)
	addRidgeToDiagInPlace(xtx, nFeat, ridge)

	xtxInv := invertSymPD(xtx, nFeat)

	xty := make([]float64, nFeat)
	applyMatVecTranspose(xty, xSub, ySub, n, nFeat)

	beta := make([]float64, nFeat)
	applyMatVec(beta, xtxInv, xty, nFeat, nFeat)

	return beta
}

/*
dotProduct computes the inner product of two equal-length vectors.
*/
func dotProduct(a, b []float64) float64 {
	return applyDotProduct(a, b)
}
