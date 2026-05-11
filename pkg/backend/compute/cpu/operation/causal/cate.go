package causal

import "fmt"

/*
CATE computes the Conditional Average Treatment Effect:
E[Y(1) - Y(0) | X=x] via outcome regression on treatment subgroups.

Fits linear models E[Y|T=1, X=x] and E[Y|T=0, X=x] separately,
then computes CATE(x) = mu_1(x) - mu_0(x) for each observation.

shape = [T, N_x, 1]
data[0] = X [T*N_x]      — covariate matrix
data[1] = T_treatment [T] — binary treatment indicator (0 or 1)
data[2] = Y [T]           — outcome vector
Returns cate [T] — individual-level CATE estimates.
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

	// Separate treated and control observations.
	treatedIdx := make([]int, 0, t)
	controlIdx := make([]int, 0, t)

	for obsIdx := 0; obsIdx < t; obsIdx++ {
		if treatment[obsIdx] >= 0.5 {
			treatedIdx = append(treatedIdx, obsIdx)
		} else {
			controlIdx = append(controlIdx, obsIdx)
		}
	}

	// Fit outcome model for treated: Y ~ X for T=1 subgroup.
	beta1 := fitOLS(xMat, yVec, treatedIdx, t, nx)

	// Fit outcome model for control: Y ~ X for T=0 subgroup.
	beta0 := fitOLS(xMat, yVec, controlIdx, t, nx)

	// CATE(x_i) = mu_1(x_i) - mu_0(x_i)
	cateValues := make([]float64, t)

	for obsIdx := 0; obsIdx < t; obsIdx++ {
		xRow := xMat[obsIdx*nx : (obsIdx+1)*nx]
		mu1 := dotProduct(beta1, xRow)
		mu0 := dotProduct(beta0, xRow)
		cateValues[obsIdx] = mu1 - mu0
	}

	return cateValues
}

/*
fitOLS fits an OLS model Y ~ X on the subset of observations given by indices.
Returns the coefficient vector beta [nx].
*/
func fitOLS(xMat, yVec []float64, indices []int, t, nx int) []float64 {
	n := len(indices)

	if n == 0 {
		return make([]float64, nx)
	}

	// Build subgroup design matrix and response.
	xSub := make([]float64, n*nx)
	ySub := make([]float64, n)

	for subIdx, obsIdx := range indices {
		copy(xSub[subIdx*nx:(subIdx+1)*nx], xMat[obsIdx*nx:(obsIdx+1)*nx])
		ySub[subIdx] = yVec[obsIdx]
	}

	// beta = (X^T X)^{-1} X^T Y
	xtx := make([]float64, nx*nx)
	applyMatMulTransposeLeft(xtx, xSub, xSub, n, nx, nx)

	xtxInv := invertSymPD(xtx, nx)

	xty := make([]float64, nx)
	applyMatVecTranspose(xty, xSub, ySub, n, nx)

	beta := make([]float64, nx)
	applyMatVec(beta, xtxInv, xty, nx, nx)

	return beta
}

/*
dotProduct computes the inner product of two equal-length vectors.
*/
func dotProduct(a, b []float64) float64 {
	return applyDotProduct(a, b)
}
