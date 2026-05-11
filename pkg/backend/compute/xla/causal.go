//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "xla_causal.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
XLACausalOps dispatches Pearl causal modeling operations to the XLA runtime via PJRT.
It provides JIT-compiled implementations of do-calculus, adjustment formulas,
IV estimation, CATE, and DAG Markov factorization on XLA-supported accelerators.
*/
type XLACausalOps struct {
	platform string
}

/*
NewCausalOps initialises the PJRT client for causal operations on the given platform.
*/
func NewCausalOps(platform string) (*XLACausalOps, error) {
	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_causal_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_causal_init failed for platform %q", platform)
	}

	return &XLACausalOps{platform: platform}, nil
}

/*
Shutdown releases all PJRT causal resources.
*/
func (xlaCausalOps *XLACausalOps) Shutdown() {
	C.xla_causal_shutdown()
}

/*
DoCalculus computes P(Y|do(X=x)) via graph surgery on a joint Gaussian.
shape=[N_vars, ...], data[0]=cov [N*N], data[1]=mask [N], data[2]=values [N]
Returns adjusted_mean [N] ++ adjusted_cov [N*N].
*/
func (xlaCausalOps *XLACausalOps) DoCalculus(shape []int, data ...[]float64) []float64 {
	n := shape[0]
	out := make([]float64, n+n*n)
	rc := C.xla_causal_do_calculus(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		panic(fmt.Sprintf("xla_causal_do_calculus failed"))
	}

	return out
}

/*
BackdoorAdjustment computes the causal effect via backdoor adjustment.
shape=[N_y, N_x, N_z, T], data[0]=Y, data[1]=X, data[2]=Z
Returns causal_effect [N_y].
*/
func (xlaCausalOps *XLACausalOps) BackdoorAdjustment(shape []int, data ...[]float64) []float64 {
	ny, nx, nz, t := shape[0], shape[1], shape[2], shape[3]
	effect := make([]float64, ny)
	rc := C.xla_causal_backdoor(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&effect[0])),
		C.int(t), C.int(ny), C.int(nx), C.int(nz),
	)

	if rc != 0 {
		panic(fmt.Sprintf("xla_causal_backdoor failed"))
	}

	return effect
}

/*
Counterfactual computes counterfactual outcomes via abduction-action-prediction.
shape=[N, N_cf], data[0]=X_obs, data[1]=Y_obs, data[2]=beta, data[3]=X_cf
Returns Y_cf [N_cf].
*/
func (xlaCausalOps *XLACausalOps) Counterfactual(shape []int, data ...[]float64) []float64 {
	n, nCF := shape[0], shape[1]
	yCF := make([]float64, nCF)
	rc := C.xla_causal_counterfactual(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&data[3][0])),
		(*C.double)(unsafe.Pointer(&yCF[0])),
		C.int(n), C.int(nCF),
	)

	if rc != 0 {
		panic(fmt.Sprintf("xla_causal_counterfactual failed"))
	}

	return yCF
}

/*
IVEstimate computes the 2SLS instrumental variable estimate.
shape=[T, N_z, N_x, N_y], data[0]=Z, data[1]=X, data[2]=Y
Returns beta_iv [N_x*N_y].
*/
func (xlaCausalOps *XLACausalOps) IVEstimate(shape []int, data ...[]float64) []float64 {
	t, nz, nx, ny := shape[0], shape[1], shape[2], shape[3]
	betaIV := make([]float64, nx*ny)
	rc := C.xla_causal_iv(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&betaIV[0])),
		C.int(t), C.int(nz), C.int(nx), C.int(ny),
	)

	if rc != 0 {
		panic(fmt.Sprintf("xla_causal_iv failed"))
	}

	return betaIV
}

/*
CATE computes the Conditional Average Treatment Effect for each observation.
shape=[T, N_x, 1], data[0]=X, data[1]=treatment, data[2]=Y
Returns cate [T].
*/
func (xlaCausalOps *XLACausalOps) CATE(shape []int, data ...[]float64) []float64 {
	t, nx := shape[0], shape[1]
	cate := make([]float64, t)
	rc := C.xla_causal_cate(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&cate[0])),
		C.int(t), C.int(nx),
	)

	if rc != 0 {
		panic(fmt.Sprintf("xla_causal_cate failed"))
	}

	return cate
}

/*
DAGMarkovFactorization computes per-observation log probabilities under the DAG Markov factorization.
shape=[N, T], data[0]=X [T*N], data[1]=adj [N*N]
Returns log_prob [T].
*/
func (xlaCausalOps *XLACausalOps) DAGMarkovFactorization(shape []int, data ...[]float64) []float64 {
	n, t := shape[0], shape[1]
	logProb := make([]float64, t)
	rc := C.xla_causal_dag_markov(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&logProb[0])),
		C.int(t), C.int(n),
	)

	if rc != 0 {
		panic(fmt.Sprintf("xla_causal_dag_markov failed"))
	}

	return logProb
}
