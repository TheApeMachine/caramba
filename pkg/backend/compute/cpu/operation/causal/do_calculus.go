package causal

import (
	"fmt"
	"math"
)

/*
DoCalculus computes P(Y|do(X=x)) via graph surgery on a joint Gaussian distribution.
Given a covariance matrix, intervention mask, and intervention values, it returns
the adjusted mean and covariance after severing incoming edges to intervened variables.

shape = [N_vars, N_vars, N_obs]
data[0] = cov_matrix [N*N]    — joint Gaussian covariance
data[1] = intervention_mask [N] — 1.0 if variable is intervened upon
data[2] = intervention_values [N] — fixed values for intervened variables
Returns adjusted_mean [N] concatenated with adjusted_cov [N*N].
*/
type DoCalculus struct{}

/*
NewDoCalculus instantiates a DoCalculus operation.
It implements Pearl's do-operator via graph surgery on Gaussian SCMs.
*/
func NewDoCalculus() *DoCalculus {
	return &DoCalculus{}
}

/*
Forward computes the post-intervention distribution P(Y|do(X=x)).
*/
func (doCalculus *DoCalculus) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Errorf("causal: DoCalculus.Forward: len(shape)=%d, need >= 2", len(shape)).Error())
	}

	if len(data) < 3 {
		panic(fmt.Errorf("causal: DoCalculus.Forward: len(data)=%d, need >= 3", len(data)).Error())
	}

	n := shape[0]

	if len(data[0]) != n*n {
		panic(fmt.Errorf(
			"causal: DoCalculus.Forward: len(data[0])=%d, need N*N=%d",
			len(data[0]), n*n,
		).Error())
	}

	if len(data[1]) != n || len(data[2]) != n {
		panic(fmt.Errorf(
			"causal: DoCalculus.Forward: mask/values len mismatch, need %d", n,
		).Error())
	}

	covMatrix := data[0]
	mask := data[1]
	values := data[2]

	// Identify intervened and free variable indices.
	intervened := make([]int, 0, n)
	free := make([]int, 0, n)

	for idx := 0; idx < n; idx++ {
		if mask[idx] != 0 {
			intervened = append(intervened, idx)
		} else {
			free = append(free, idx)
		}
	}

	// Graph surgery: set intervened variables to fixed values.
	// Under do(X=x), the joint distribution factorizes such that
	// intervened variables become constants (delta distributions).
	// For the free variables, we condition on the intervened values
	// using the Gaussian conditional formula.
	adjustedMean := make([]float64, n)
	adjustedCov := make([]float64, n*n)

	// Copy original covariance — we will zero out rows/cols of intervened vars.
	copy(adjustedCov, covMatrix)

	// For intervened variables: mean = intervention value, variance = 0.
	for _, interventionIdx := range intervened {
		adjustedMean[interventionIdx] = values[interventionIdx]
	}

	// For free variables: compute conditional mean given intervened values.
	// Using block Gaussian conditioning: mu_free|do = mu_free + Sigma_{free,int} * Sigma_{int,int}^{-1} * (x_int - mu_int)
	// Since we assume zero-mean SCM (mean=0 prior to intervention), simplifies to:
	// mu_free|do = Sigma_{free,int} * Sigma_{int,int}^{-1} * x_int
	if len(intervened) > 0 && len(free) > 0 {
		ni := len(intervened)
		nf := len(free)

		// Extract Sigma_int_int [ni x ni]
		sigIntInt := make([]float64, ni*ni)

		for row, ii := range intervened {
			for col, jj := range intervened {
				sigIntInt[row*ni+col] = covMatrix[ii*n+jj]
			}
		}

		// Extract Sigma_free_int [nf x ni]
		sigFreeInt := make([]float64, nf*ni)

		for row, fi := range free {
			for col, ii := range intervened {
				sigFreeInt[row*ni+col] = covMatrix[fi*n+ii]
			}
		}

		// Invert Sigma_int_int.
		sigIntIntInv := invertSymPD(sigIntInt, ni)

		// x_int vector.
		xInt := make([]float64, ni)

		for idx, ii := range intervened {
			xInt[idx] = values[ii]
		}

		// tmp = Sigma_int_int_inv @ x_int  [ni]
		tmp := make([]float64, ni)
		applyMatVec(tmp, sigIntIntInv, xInt, ni, ni)

		// delta_free = Sigma_free_int @ tmp  [nf]
		deltaFree := make([]float64, nf)
		applyMatVec(deltaFree, sigFreeInt, tmp, nf, ni)

		for idx, fi := range free {
			adjustedMean[fi] = deltaFree[idx]
		}

		// Conditional covariance: Sigma_free|int = Sigma_free_free - Sigma_free_int @ Sigma_int_int_inv @ Sigma_int_free
		// Sigma_free_free [nf x nf]
		sigFreeFree := make([]float64, nf*nf)

		for row, fi := range free {
			for col, fj := range free {
				sigFreeFree[row*nf+col] = covMatrix[fi*n+fj]
			}
		}

		// tmp2 = Sigma_int_int_inv @ Sigma_int_free  [ni x nf]
		sigIntFree := make([]float64, ni*nf)

		for row, ii := range intervened {
			for col, fj := range free {
				sigIntFree[row*nf+col] = covMatrix[ii*n+fj]
			}
		}

		tmp2 := make([]float64, ni*nf)
		applyMatMulFull(tmp2, sigIntIntInv, sigIntFree, ni, ni, nf)

		// correction = Sigma_free_int @ tmp2  [nf x nf]
		correction := make([]float64, nf*nf)
		applyMatMulFull(correction, sigFreeInt, tmp2, nf, ni, nf)

		// Place corrected covariance back into adjustedCov for free vars.
		for row, fi := range free {
			for col, fj := range free {
				adjustedCov[fi*n+fj] = sigFreeFree[row*nf+col] - correction[row*nf+col]
			}
		}
	}

	// Zero out covariance rows/cols for intervened variables.
	for _, interventionIdx := range intervened {
		for j := 0; j < n; j++ {
			adjustedCov[interventionIdx*n+j] = 0
			adjustedCov[j*n+interventionIdx] = 0
		}
	}

	result := make([]float64, n+n*n)
	copy(result[:n], adjustedMean)
	copy(result[n:], adjustedCov)

	return result
}

/*
invertSymPD inverts a symmetric positive-definite matrix using Cholesky decomposition.
Operates on a flat row-major [n*n] slice.
*/
func invertSymPD(a []float64, n int) []float64 {
	// Cholesky: A = L L^T
	lower := make([]float64, n*n)

	for row := 0; row < n; row++ {
		for col := 0; col <= row; col++ {
			sum := a[row*n+col]

			for k := 0; k < col; k++ {
				sum -= lower[row*n+k] * lower[col*n+k]
			}

			if row == col {
				if sum <= 0 {
					// Not positive definite — add regularization.
					sum = 1e-10
				}

				lower[row*n+col] = math.Sqrt(sum)
			} else {
				lower[row*n+col] = sum / lower[col*n+col]
			}
		}
	}

	// Invert via forward/back substitution: solve L * Y = I, then L^T * X = Y.
	inv := make([]float64, n*n)

	for col := 0; col < n; col++ {
		// Forward substitution for column col of identity.
		y := make([]float64, n)
		y[col] = 1.0

		for row := col; row < n; row++ {
			if row > col {
				y[row] = 0

				for k := col; k < row; k++ {
					y[row] -= lower[row*n+k] * y[k]
				}
			}

			y[row] /= lower[row*n+row]
		}

		// Back substitution.
		x := make([]float64, n)

		for row := n - 1; row >= 0; row-- {
			x[row] = y[row]

			for k := row + 1; k < n; k++ {
				x[row] -= lower[k*n+row] * x[k]
			}

			x[row] /= lower[row*n+row]
		}

		for row := 0; row < n; row++ {
			inv[row*n+col] = x[row]
		}
	}

	return inv
}
