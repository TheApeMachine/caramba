//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "causal.h"
import "C"

import (
	"fmt"
	"strings"
	"sync"
	"unsafe"
)

/*
MetalCausalOps dispatches Pearl causal modeling on the GPU (causal.metallib).

Calls are serialised by a mutex so one instance may be used from multiple goroutines.
*/
type MetalCausalOps struct {
	mu       sync.Mutex
	metallib string
}

/*
NewCausalOps creates and initialises a MetalCausalOps instance.
*/
func NewCausalOps(metallib string) (*MetalCausalOps, error) {
	if strings.TrimSpace(metallib) == "" {
		return nil, fmt.Errorf("NewCausalOps: metallib path is empty")
	}

	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_causal_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_causal_init failed (rc=%d): check %q exists", rc, metallib)
	}

	return &MetalCausalOps{metallib: metallib}, nil
}

/*
Close releases resources from metal_causal_init.
*/
func (metalCausalOps *MetalCausalOps) Close() error {
	metalCausalOps.mu.Lock()
	defer metalCausalOps.mu.Unlock()

	if rc := C.metal_causal_shutdown(); rc != 0 {
		return fmt.Errorf("metal_causal_shutdown failed (rc=%d)", rc)
	}

	return nil
}

/*
DoCalculus computes P(Y|do(X=x)) via graph surgery on a joint Gaussian.
shape=[N_vars, ...], data[0]=cov [N*N], data[1]=mask [N], data[2]=values [N]
Returns adjusted_mean [N] ++ adjusted_cov [N*N].
*/
func (metalCausalOps *MetalCausalOps) DoCalculus(shape []int, data ...[]float64) ([]float64, error) {
	metalCausalOps.mu.Lock()
	defer metalCausalOps.mu.Unlock()

	if len(shape) < 1 {
		return nil, fmt.Errorf("MetalCausalOps.DoCalculus: need len(shape) >= 1")
	}

	n := shape[0]

	if n <= 0 {
		return nil, fmt.Errorf("MetalCausalOps.DoCalculus: need N > 0")
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("MetalCausalOps.DoCalculus: need len(data) >= 3")
	}

	if len(data[0]) != n*n || len(data[1]) != n || len(data[2]) != n {
		return nil, fmt.Errorf("MetalCausalOps.DoCalculus: need len(cov)=%d, len(mask|values)=%d", n*n, n)
	}

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
	metalCausalOps.mu.Lock()
	defer metalCausalOps.mu.Unlock()

	if len(shape) < 4 {
		return nil, fmt.Errorf("MetalCausalOps.BackdoorAdjustment: need len(shape) >= 4")
	}

	ny, nx, nz, t := shape[0], shape[1], shape[2], shape[3]

	if ny <= 0 || nx <= 0 || t <= 0 || nz < 0 {
		return nil, fmt.Errorf("MetalCausalOps.BackdoorAdjustment: invalid dimensions")
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("MetalCausalOps.BackdoorAdjustment: need len(data) >= 3")
	}

	expX := t * nx
	expY := t * ny
	expZ := t * nz

	if len(data[0]) < expY || len(data[1]) < expX || len(data[2]) < expZ {
		return nil, fmt.Errorf(
			"MetalCausalOps.BackdoorAdjustment: need len(Y)>=%d, len(X)>=%d, len(Z)>=%d",
			expY, expX, expZ,
		)
	}

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
	metalCausalOps.mu.Lock()
	defer metalCausalOps.mu.Unlock()

	if len(shape) < 4 {
		return nil, fmt.Errorf("MetalCausalOps.IVEstimate: need len(shape) >= 4")
	}

	t, nz, nx, ny := shape[0], shape[1], shape[2], shape[3]

	if t <= 0 || nz <= 0 || nx <= 0 || ny <= 0 {
		return nil, fmt.Errorf("MetalCausalOps.IVEstimate: dimensions must be positive")
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("MetalCausalOps.IVEstimate: need len(data) >= 3")
	}

	expZ := t * nz
	expX := t * nx
	expY := t * ny

	if len(data[0]) < expZ || len(data[1]) < expX || len(data[2]) < expY {
		return nil, fmt.Errorf(
			"MetalCausalOps.IVEstimate: need len(Z)>=%d, len(X)>=%d, len(Y)>=%d",
			expZ, expX, expY,
		)
	}

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
	metalCausalOps.mu.Lock()
	defer metalCausalOps.mu.Unlock()

	if len(shape) < 2 {
		return nil, fmt.Errorf("MetalCausalOps.CATE: need len(shape) >= 2")
	}

	t, nx := shape[0], shape[1]

	if t <= 0 || nx <= 0 {
		return nil, fmt.Errorf("MetalCausalOps.CATE: need T>0 and N_x>0")
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("MetalCausalOps.CATE: need len(data) >= 3")
	}

	if len(data[0]) < t*nx || len(data[1]) < t || len(data[2]) < t {
		return nil, fmt.Errorf("MetalCausalOps.CATE: slice lengths too short for T=%d, nx=%d", t, nx)
	}

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
	metalCausalOps.mu.Lock()
	defer metalCausalOps.mu.Unlock()

	if len(shape) < 2 {
		return nil, fmt.Errorf("MetalCausalOps.DAGMarkovFactorization: need len(shape) >= 2")
	}

	n, t := shape[0], shape[1]

	if n <= 0 || t <= 0 {
		return nil, fmt.Errorf("MetalCausalOps.DAGMarkovFactorization: need N>0 and T>0")
	}

	if len(data) < 2 {
		return nil, fmt.Errorf("MetalCausalOps.DAGMarkovFactorization: need len(data) >= 2")
	}

	if len(data[0]) < t*n || len(data[1]) < n*n {
		return nil, fmt.Errorf(
			"MetalCausalOps.DAGMarkovFactorization: need len(X)>=%d, len(adj)>=%d",
			t*n, n*n,
		)
	}

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
