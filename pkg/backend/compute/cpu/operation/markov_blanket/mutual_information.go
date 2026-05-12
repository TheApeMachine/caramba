package markov_blanket

import (
	"fmt"

	mathops "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

/*
MutualInformation estimates mutual information between two multivariate
Gaussian-approximated marginals X and Y via:

	MI = 0.5 * log( det(Σ_x) * det(Σ_y) / det(Σ_joint) )
	   = 0.5 * (logdet(Σ_x) + logdet(Σ_y) - logdet(Σ_joint))

shape = [N, M]
data[0] = X [T * N]  (T samples, N dims)
data[1] = Y [T * M]  (T samples, M dims)

Returns scalar MI estimate as []float64{mi}.
*/
type MutualInformation struct{}

func NewMutualInformation() *MutualInformation { return &MutualInformation{} }

func (op *MutualInformation) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Errorf("markov_blanket: MutualInformation: len(shape)=%d, need 2", len(shape)).Error())
	}

	if len(data) < 2 {
		panic(fmt.Errorf("markov_blanket: MutualInformation: len(data)=%d, need 2", len(data)).Error())
	}

	N, M := shape[0], shape[1]
	xData := data[0]
	yData := data[1]

	if len(xData)%N != 0 {
		panic(fmt.Errorf(
			"markov_blanket: MutualInformation: len(X)=%d not divisible by N=%d",
			len(xData), N,
		).Error())
	}

	T := len(xData) / N

	if T < 2 {
		panic(fmt.Sprintf(
			"markov_blanket: MutualInformation: need at least 2 samples (T>=2) before computeMI(xData, yData, T, N, M); got T=%d len(xData)=%d len(yData)=%d N=%d M=%d",
			T, len(xData), len(yData), N, M,
		))
	}

	if len(yData) != T*M {
		panic(fmt.Errorf(
			"markov_blanket: MutualInformation: len(Y)=%d, need T*M=%d",
			len(yData), T*M,
		).Error())
	}

	mi := computeMI(xData, yData, T, N, M)

	return []float64{mi}
}

// computeMI estimates MI using the Gaussian approximation via log-det ratio.
func computeMI(xData, yData []float64, T, N, M int) float64 {
	xMean := columnMean(xData, T, N)
	yMean := columnMean(yData, T, M)

	covX := covariance(xData, xMean, T, N)
	covY := covariance(yData, yMean, T, M)
	covJoint := crossCovariance(xData, yData, xMean, yMean, T, N, M)

	// Build joint covariance matrix [N+M, N+M]
	NM := N + M
	sigmaJoint := make([]float64, NM*NM)

	for row := range N {
		for col := range N {
			sigmaJoint[row*NM+col] = covX[row*N+col]
		}
	}

	for row := range M {
		for col := range M {
			sigmaJoint[(N+row)*NM+(N+col)] = covY[row*M+col]
		}
	}

	for row := range N {
		for col := range M {
			sigmaJoint[row*NM+(N+col)] = covJoint[row*M+col]
			sigmaJoint[(N+col)*NM+row] = covJoint[row*M+col]
		}
	}

	ldX := logDetCholesky(covX, N)
	ldY := logDetCholesky(covY, M)
	ldJ := logDetCholesky(sigmaJoint, NM)

	mi := 0.5 * (ldX + ldY - ldJ)

	if mi < 0 {
		mi = 0
	}

	return mi
}

func columnMean(data []float64, T, D int) []float64 {
	mean := make([]float64, D)

	if T <= 0 {
		return mean
	}

	for t := range T {
		for d := range D {
			mean[d] += data[t*D+d]
		}
	}

	invT := 1.0 / float64(T)

	for d := range D {
		mean[d] *= invT
	}

	return mean
}

func covariance(data, mean []float64, T, D int) []float64 {
	cov := make([]float64, D*D)
	invT := 1.0 / float64(T-1)

	for t := range T {
		for row := range D {
			dr := data[t*D+row] - mean[row]

			for col := row; col < D; col++ {
				dc := data[t*D+col] - mean[col]
				cov[row*D+col] += dr * dc
			}
		}
	}

	for row := range D {
		for col := row; col < D; col++ {
			cov[row*D+col] *= invT
			cov[col*D+row] = cov[row*D+col]
		}
	}

	return cov
}

func crossCovariance(xData, yData, xMean, yMean []float64, T, N, M int) []float64 {
	cov := make([]float64, N*M)
	invT := 1.0 / float64(T-1)

	for t := range T {
		for row := range N {
			dr := xData[t*N+row] - xMean[row]

			for col := range M {
				cov[row*M+col] += dr * (yData[t*M+col] - yMean[col])
			}
		}
	}

	for idx := range cov {
		cov[idx] *= invT
	}

	return cov
}

// logDetCholesky computes log|A| via Cholesky decomposition (A must be PSD).
// Regularises A with a small diagonal term for numerical stability.
func logDetCholesky(a []float64, n int) float64 {
	// Copy with diagonal regularisation.
	L := make([]float64, n*n)
	copy(L, a)

	eps := 1e-10

	for diag := range n {
		L[diag*n+diag] += eps
	}

	if !choleskyDecomp(L, n) {
		panic(fmt.Sprintf(
			"markov_blanket: logDetCholesky: non-positive pivot during decomposition (n=%d); matrix may be non-PD after regularisation",
			n,
		))
	}

	// log|A| = 2 * sum_i log(L[i,i]) — gather diagonal then SIMD log+sum.
	diag := make([]float64, n)

	for idx := range n {
		diag[idx] = L[idx*n+idx]
	}

	mathops.LogVec(diag, diag)

	return 2.0 * mathops.ReduceSum(diag)
}
