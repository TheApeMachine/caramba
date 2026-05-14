//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "causal.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
CUDACausalOps dispatches Pearl causal modeling kernels to the GPU via CUDA.
It provides GPU-accelerated implementations of do-calculus, backdoor/frontdoor
adjustment, IV estimation, CATE, and DAG Markov factorization.
*/
type CUDACausalOps struct{}

/*
NewCausalOps creates a CUDACausalOps instance.
*/
func NewCausalOps() *CUDACausalOps {
	return &CUDACausalOps{}
}

/*
DoCalculus computes P(Y|do(X=x)) via graph surgery on a joint Gaussian.
shape=[N_vars, ...], data[0]=cov [N*N], data[1]=mask [N], data[2]=values [N]
Returns adjusted_mean [N] + adjusted_cov [N*N].
*/
func (cudaCausalOps *CUDACausalOps) DoCalculus(shape []int, data ...[]float64) ([]float64, error) {
	n := shape[0]
	out := make([]float64, n+n*n)
	rc := C.cuda_causal_do_calculus(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_causal_do_calculus failed (rc=%d)", rc)
	}

	return out, nil
}

/*
BackdoorAdjustment computes the causal effect via backdoor adjustment.
shape=[N_y, N_x, N_z, T], data[0]=Y, data[1]=X, data[2]=Z
Returns causal_effect [N_y].
*/
func (cudaCausalOps *CUDACausalOps) BackdoorAdjustment(
	shape []int, data ...[]float64,
) ([]float64, error) {
	ny, nx, nz, t := shape[0], shape[1], shape[2], shape[3]
	effect := make([]float64, ny)
	rc := C.cuda_causal_backdoor(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&effect[0])),
		C.int(t), C.int(ny), C.int(nx), C.int(nz),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_causal_backdoor failed (rc=%d)", rc)
	}

	return effect, nil
}

/*
IVEstimate computes the 2SLS instrumental variable estimate.
shape=[T, N_z, N_x, N_y], data[0]=Z, data[1]=X, data[2]=Y
Returns beta_iv [N_x*N_y].
*/
func (cudaCausalOps *CUDACausalOps) IVEstimate(
	shape []int, data ...[]float64,
) ([]float64, error) {
	t, nz, nx, ny := shape[0], shape[1], shape[2], shape[3]
	betaIV := make([]float64, nx*ny)
	rc := C.cuda_causal_iv(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&betaIV[0])),
		C.int(t), C.int(nz), C.int(nx), C.int(ny),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_causal_iv failed (rc=%d)", rc)
	}

	return betaIV, nil
}

/*
CATE computes the Conditional Average Treatment Effect for each observation.
shape=[T, N_x, 1], data[0]=X, data[1]=treatment, data[2]=Y
Returns cate [T].
*/
func (cudaCausalOps *CUDACausalOps) CATE(shape []int, data ...[]float64) ([]float64, error) {
	t, nx := shape[0], shape[1]
	cate := make([]float64, t)
	rc := C.cuda_causal_cate(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&cate[0])),
		C.int(t), C.int(nx),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_causal_cate failed (rc=%d)", rc)
	}

	return cate, nil
}

/*
DAGMarkovFactorization computes per-observation log probabilities under the DAG Markov factorization.
shape=[N, T], data[0]=X [T*N], data[1]=adj [N*N]
Returns log_prob [T].
*/
func (cudaCausalOps *CUDACausalOps) DAGMarkovFactorization(
	shape []int, data ...[]float64,
) ([]float64, error) {
	n, t := shape[0], shape[1]
	logProb := make([]float64, t)
	rc := C.cuda_causal_dag_markov(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&logProb[0])),
		C.int(t), C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_causal_dag_markov failed (rc=%d)", rc)
	}

	return logProb, nil
}

/*
Counterfactual runs a heterogeneous linear SCM counterfactual query.
shape=[N, N_cf], data[0]=X_obs, data[1]=Y_obs, data[2]=beta, data[3]=X_cf.
Returns counterfactual_out [N*N_cf], where each block describes the predicted
counterfactual outcome for observed unit N under candidate X_cf.
*/
func (cudaCausalOps *CUDACausalOps) Counterfactual(
	shape []int, data ...[]float64,
) ([]float64, error) {
	n, nCF := shape[0], shape[1]
	output := make([]float64, n*nCF)
	rc := C.cuda_causal_counterfactual(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&data[3][0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int(n), C.int(nCF),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_causal_counterfactual failed (rc=%d)", rc)
	}

	return output, nil
}

/*
FrontdoorAdjustment computes the frontdoor causal effect with equal-frequency binning.
shape=[N_x, N_m, N_y, T], data[0]=X, data[1]=M, data[2]=Y.
N_y is retained for API symmetry and must be 1 because the CUDA kernel models
univariate Y. Returns effect [N_x], a float64 slice of length N_x containing the
estimated frontdoor causal effect per X bin.
*/
func (cudaCausalOps *CUDACausalOps) FrontdoorAdjustment(
	shape []int, data ...[]float64,
) ([]float64, error) {
	if len(shape) < 4 {
		return nil, fmt.Errorf("cuda_causal_frontdoor: shape=[N_x,N_m,N_y,T] is required")
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("cuda_causal_frontdoor: X, M, and Y inputs are required")
	}

	nx, nm, ny, samples := shape[0], shape[1], shape[2], shape[3]

	if nx <= 0 || nm <= 0 || samples <= 0 {
		return nil, fmt.Errorf("cuda_causal_frontdoor: N_x, N_m, and T must be positive")
	}

	if ny != 1 {
		return nil, fmt.Errorf("cuda_causal_frontdoor: N_y must be 1 for univariate Y, got %d", ny)
	}

	effect := make([]float64, nx)
	rc := C.cuda_causal_frontdoor(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&effect[0])),
		C.int(samples), C.int(nx), C.int(nm),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_causal_frontdoor failed (rc=%d)", rc)
	}

	return effect, nil
}
