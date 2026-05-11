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
func (cudaCausalOps *CUDACausalOps) DoCalculus(shape []int, data ...[]float64) []float64 {
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
		panic(fmt.Sprintf("cuda_causal_do_calculus failed (rc=%d)", rc))
	}

	return out
}

/*
BackdoorAdjustment computes the causal effect via backdoor adjustment.
shape=[N_y, N_x, N_z, T], data[0]=Y, data[1]=X, data[2]=Z
Returns causal_effect [N_y].
*/
func (cudaCausalOps *CUDACausalOps) BackdoorAdjustment(shape []int, data ...[]float64) []float64 {
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
		panic(fmt.Sprintf("cuda_causal_backdoor failed (rc=%d)", rc))
	}

	return effect
}

/*
IVEstimate computes the 2SLS instrumental variable estimate.
shape=[T, N_z, N_x, N_y], data[0]=Z, data[1]=X, data[2]=Y
Returns beta_iv [N_x*N_y].
*/
func (cudaCausalOps *CUDACausalOps) IVEstimate(shape []int, data ...[]float64) []float64 {
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
		panic(fmt.Sprintf("cuda_causal_iv failed (rc=%d)", rc))
	}

	return betaIV
}

/*
CATE computes the Conditional Average Treatment Effect for each observation.
shape=[T, N_x, 1], data[0]=X, data[1]=treatment, data[2]=Y
Returns cate [T].
*/
func (cudaCausalOps *CUDACausalOps) CATE(shape []int, data ...[]float64) []float64 {
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
		panic(fmt.Sprintf("cuda_causal_cate failed (rc=%d)", rc))
	}

	return cate
}

/*
DAGMarkovFactorization computes per-observation log probabilities under the DAG Markov factorization.
shape=[N, T], data[0]=X [T*N], data[1]=adj [N*N]
Returns log_prob [T].
*/
func (cudaCausalOps *CUDACausalOps) DAGMarkovFactorization(shape []int, data ...[]float64) []float64 {
	n, t := shape[0], shape[1]
	logProb := make([]float64, t)
	rc := C.cuda_causal_dag_markov(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&logProb[0])),
		C.int(t), C.int(n),
	)

	if rc != 0 {
		panic(fmt.Sprintf("cuda_causal_dag_markov failed (rc=%d)", rc))
	}

	return logProb
}
