//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include <math.h>
// #include <string.h>
// #include "causal.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
MetalCausalOps dispatches Pearl causal modeling operations to the GPU via Metal.
It provides GPU-accelerated do-calculus, adjustment formulas, IV estimation,
CATE computation, and DAG Markov factorization for Apple Silicon and AMD GPUs.
*/
type MetalCausalOps struct {
	metallib string
}

/*
NewCausalOps creates and initialises a MetalCausalOps instance.
*/
func NewCausalOps(metallib string) (*MetalCausalOps, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_causal_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_causal_init failed (rc=%d): check %q exists", rc, metallib)
	}

	return &MetalCausalOps{metallib: metallib}, nil
}

/*
DoCalculus computes P(Y|do(X=x)) via graph surgery on a joint Gaussian.
shape=[N_vars, ...], data[0]=cov [N*N], data[1]=mask [N], data[2]=values [N]
Returns adjusted_mean [N] ++ adjusted_cov [N*N].
*/
func (metalCausalOps *MetalCausalOps) DoCalculus(shape []int, data ...[]float64) ([]float64, error) {
	n := shape[0]
	cov := toFloat32(data[0])
	mask := toFloat32(data[1])
	values := toFloat32(data[2])
	out := make([]float32, n+n*n)
	rc := C.metal_causal_do_calculus(
		(*C.float)(unsafe.Pointer(&cov[0])),
		(*C.float)(unsafe.Pointer(&mask[0])),
		(*C.float)(unsafe.Pointer(&values[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_causal_do_calculus failed (rc=%d)", rc)
	}

	return toFloat64(out), nil
}

/*
BackdoorAdjustment computes the causal effect via backdoor adjustment.
shape=[N_y, N_x, N_z, T], data[0]=Y, data[1]=X, data[2]=Z
Returns causal_effect [N_y].
*/
func (metalCausalOps *MetalCausalOps) BackdoorAdjustment(shape []int, data ...[]float64) ([]float64, error) {
	ny, nx, nz, t := shape[0], shape[1], shape[2], shape[3]
	yF := toFloat32(data[0])
	xF := toFloat32(data[1])
	zF := toFloat32(data[2])
	effect := make([]float32, ny)
	rc := C.metal_causal_backdoor(
		(*C.float)(unsafe.Pointer(&yF[0])),
		(*C.float)(unsafe.Pointer(&xF[0])),
		(*C.float)(unsafe.Pointer(&zF[0])),
		(*C.float)(unsafe.Pointer(&effect[0])),
		C.int(t), C.int(ny), C.int(nx), C.int(nz),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_causal_backdoor failed (rc=%d)", rc)
	}

	return toFloat64(effect), nil
}

/*
IVEstimate computes the 2SLS instrumental variable estimate.
shape=[T, N_z, N_x, N_y], data[0]=Z, data[1]=X, data[2]=Y
Returns beta_iv [N_x*N_y].
*/
func (metalCausalOps *MetalCausalOps) IVEstimate(shape []int, data ...[]float64) ([]float64, error) {
	t, nz, nx, ny := shape[0], shape[1], shape[2], shape[3]
	zF := toFloat32(data[0])
	xF := toFloat32(data[1])
	yF := toFloat32(data[2])
	betaIV := make([]float32, nx*ny)
	rc := C.metal_causal_iv(
		(*C.float)(unsafe.Pointer(&zF[0])),
		(*C.float)(unsafe.Pointer(&xF[0])),
		(*C.float)(unsafe.Pointer(&yF[0])),
		(*C.float)(unsafe.Pointer(&betaIV[0])),
		C.int(t), C.int(nz), C.int(nx), C.int(ny),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_causal_iv failed (rc=%d)", rc)
	}

	return toFloat64(betaIV), nil
}

/*
CATE computes the Conditional Average Treatment Effect for each observation.
shape=[T, N_x, 1], data[0]=X, data[1]=treatment, data[2]=Y
Returns cate [T].
*/
func (metalCausalOps *MetalCausalOps) CATE(shape []int, data ...[]float64) ([]float64, error) {
	t, nx := shape[0], shape[1]
	xF := toFloat32(data[0])
	treatF := toFloat32(data[1])
	yF := toFloat32(data[2])
	cate := make([]float32, t)
	rc := C.metal_causal_cate(
		(*C.float)(unsafe.Pointer(&xF[0])),
		(*C.float)(unsafe.Pointer(&treatF[0])),
		(*C.float)(unsafe.Pointer(&yF[0])),
		(*C.float)(unsafe.Pointer(&cate[0])),
		C.int(t), C.int(nx),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_causal_cate failed (rc=%d)", rc)
	}

	return toFloat64(cate), nil
}

/*
DAGMarkovFactorization computes per-observation log probabilities under the DAG Markov factorization.
shape=[N, T], data[0]=X [T*N], data[1]=adj [N*N]
Returns log_prob [T].
*/
func (metalCausalOps *MetalCausalOps) DAGMarkovFactorization(shape []int, data ...[]float64) ([]float64, error) {
	n, t := shape[0], shape[1]
	xF := toFloat32(data[0])
	adjF := toFloat32(data[1])
	logProb := make([]float32, t)
	rc := C.metal_causal_dag_markov(
		(*C.float)(unsafe.Pointer(&xF[0])),
		(*C.float)(unsafe.Pointer(&adjF[0])),
		(*C.float)(unsafe.Pointer(&logProb[0])),
		C.int(t), C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_causal_dag_markov failed (rc=%d)", rc)
	}

	return toFloat64(logProb), nil
}
