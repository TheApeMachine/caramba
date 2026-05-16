//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "xla_causal.h"
import "C"

import (
	"fmt"
	"math"
	"sync"
	"unsafe"
)

/*
XLACausalOps dispatches Pearl causal modeling operations to the XLA runtime via PJRT.
*/
type XLACausalOps struct {
	mu     sync.RWMutex
	closed bool
}

/*
NewCausalOps initialises the PJRT client for causal operations on the given platform.
Platform is validated via PJRTConfig (cpu, gpu, cuda).
*/
func NewCausalOps(platform string) (*XLACausalOps, error) {
	config, err := newRuntimePJRTConfig(platform)

	if err != nil {
		return nil, err
	}

	cp := C.CString(config.Platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_causal_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_causal_init failed for platform %q (rc=%d)", config.Platform, rc)
	}

	return &XLACausalOps{}, nil
}

func (xlaCausalOps *XLACausalOps) errIfClosed(name string) error {
	if xlaCausalOps.closed {
		return fmt.Errorf("%s: XLACausalOps already shut down", name)
	}

	return nil
}

func cIntCausal(name string, v int) (C.int, error) {
	if v < 0 || v > math.MaxInt32 {
		return 0, fmt.Errorf("%s: value %d out of range for C.int", name, v)
	}

	return C.int(int32(v)), nil
}

/*
Shutdown releases all PJRT causal resources.
*/
func (xlaCausalOps *XLACausalOps) Shutdown() {
	xlaCausalOps.mu.Lock()
	defer xlaCausalOps.mu.Unlock()

	if xlaCausalOps.closed {
		return
	}

	xlaCausalOps.closed = true
	C.xla_causal_shutdown()
}

/*
DoCalculus computes P(Y|do(X=x)) via graph surgery on a joint Gaussian.
shape=[N_vars, ...], data[0]=cov [N*N], data[1]=mask [N], data[2]=values [N]
Returns adjusted_mean [N] ++ adjusted_cov [N*N].
*/
func (xlaCausalOps *XLACausalOps) DoCalculus(shape []int, data ...[]float64) ([]float64, error) {
	xlaCausalOps.mu.RLock()
	defer xlaCausalOps.mu.RUnlock()

	if err := xlaCausalOps.errIfClosed("DoCalculus"); err != nil {
		return nil, err
	}

	if len(shape) < 1 {
		return nil, fmt.Errorf("DoCalculus: len(shape) < 1")
	}

	n := shape[0]

	if n <= 0 || n > math.MaxInt32 {
		return nil, fmt.Errorf("DoCalculus: invalid n=%d", n)
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("DoCalculus: len(data) < 3")
	}

	if len(data[0]) < n*n || len(data[1]) < n || len(data[2]) < n {
		return nil, fmt.Errorf("DoCalculus: data slice lengths mismatch for n=%d", n)
	}

	cn, err := cIntCausal("DoCalculus.n", n)
	if err != nil {
		return nil, err
	}

	out := make([]float64, n+n*n)
	rc := C.xla_causal_do_calculus(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		cn,
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_causal_do_calculus failed (rc=%d)", rc)
	}

	return out, nil
}

/*
BackdoorAdjustment computes the causal effect via backdoor adjustment.
shape=[N_y, N_x, N_z, T], data[0]=Y, data[1]=X, data[2]=Z
Returns causal_effect [N_y].
*/
func (xlaCausalOps *XLACausalOps) BackdoorAdjustment(shape []int, data ...[]float64) ([]float64, error) {
	xlaCausalOps.mu.RLock()
	defer xlaCausalOps.mu.RUnlock()

	if err := xlaCausalOps.errIfClosed("BackdoorAdjustment"); err != nil {
		return nil, err
	}

	if len(shape) < 4 {
		return nil, fmt.Errorf("BackdoorAdjustment: len(shape) < 4")
	}

	ny, nx, nz, t := shape[0], shape[1], shape[2], shape[3]

	if ny <= 0 || nx <= 0 || nz < 0 || t <= 0 {
		return nil, fmt.Errorf("BackdoorAdjustment: non-positive dimension ny=%d nx=%d nz=%d t=%d", ny, nx, nz, t)
	}

	if ny > math.MaxInt32 || nx > math.MaxInt32 || nz > math.MaxInt32 || t > math.MaxInt32 {
		return nil, fmt.Errorf("BackdoorAdjustment: dimension out of range for C.int")
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("BackdoorAdjustment: len(data) < 3")
	}

	if len(data[0]) < ny*t || len(data[1]) < nx*t {
		return nil, fmt.Errorf("BackdoorAdjustment: data slice length mismatch (ny*t=%d nx*t=%d)",
			ny*t, nx*t)
	}

	if nz > 0 && len(data[2]) < nz*t {
		return nil, fmt.Errorf("BackdoorAdjustment: Z slice length mismatch nz*t=%d", nz*t)
	}

	ct, err := cIntCausal("BackdoorAdjustment.t", t)
	if err != nil {
		return nil, err
	}

	cny, err := cIntCausal("BackdoorAdjustment.ny", ny)
	if err != nil {
		return nil, err
	}

	cnx, err := cIntCausal("BackdoorAdjustment.nx", nx)
	if err != nil {
		return nil, err
	}

	cnz, err := cIntCausal("BackdoorAdjustment.nz", nz)
	if err != nil {
		return nil, err
	}

	effect := make([]float64, ny)
	var zPtr *C.double
	if nz > 0 {
		zPtr = (*C.double)(unsafe.Pointer(&data[2][0]))
	}

	rc := C.xla_causal_backdoor(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		zPtr,
		(*C.double)(unsafe.Pointer(&effect[0])),
		ct, cny, cnx, cnz,
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_causal_backdoor failed (rc=%d)", rc)
	}

	return effect, nil
}

/*
FrontdoorAdjustment computes the frontdoor-identified causal effect (equal-frequency binning).
shape=[N_x, N_m, N_y, T] (N_y reserved for parity with CPU API), data[0]=X[T], data[1]=M[T], data[2]=Y[T].
Returns causal_effect [N_x].
*/
func (xlaCausalOps *XLACausalOps) FrontdoorAdjustment(shape []int, data ...[]float64) ([]float64, error) {
	xlaCausalOps.mu.RLock()
	defer xlaCausalOps.mu.RUnlock()

	if err := xlaCausalOps.errIfClosed("FrontdoorAdjustment"); err != nil {
		return nil, err
	}

	if len(shape) < 4 {
		return nil, fmt.Errorf("FrontdoorAdjustment: len(shape) < 4")
	}

	nx, nm, _, t := shape[0], shape[1], shape[2], shape[3]

	if nx <= 0 || nm <= 0 || t <= 0 {
		return nil, fmt.Errorf("FrontdoorAdjustment: invalid nx=%d nm=%d t=%d", nx, nm, t)
	}

	if nx > math.MaxInt32 || nm > math.MaxInt32 || t > math.MaxInt32 {
		return nil, fmt.Errorf("FrontdoorAdjustment: dimension out of range for C.int")
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("FrontdoorAdjustment: len(data) < 3")
	}

	if len(data[0]) < t || len(data[1]) < t || len(data[2]) < t {
		return nil, fmt.Errorf("FrontdoorAdjustment: data lengths must be >= T=%d", t)
	}

	ct, err := cIntCausal("FrontdoorAdjustment.t", t)
	if err != nil {
		return nil, err
	}

	cnx, err := cIntCausal("FrontdoorAdjustment.nx", nx)
	if err != nil {
		return nil, err
	}

	cnm, err := cIntCausal("FrontdoorAdjustment.nm", nm)
	if err != nil {
		return nil, err
	}

	effect := make([]float64, nx)
	rc := C.xla_causal_frontdoor(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&effect[0])),
		ct, cnx, cnm,
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_causal_frontdoor failed (rc=%d)", rc)
	}

	return effect, nil
}

/*
Counterfactual computes counterfactual outcomes via abduction-action-prediction.
shape=[N, N_cf], data[0]=X_obs, data[1]=Y_obs, data[2]=beta, data[3]=X_cf
Returns Y_cf flat row-major [N × N_cf] of length N*N_cf.
*/
func (xlaCausalOps *XLACausalOps) Counterfactual(shape []int, data ...[]float64) ([]float64, error) {
	xlaCausalOps.mu.RLock()
	defer xlaCausalOps.mu.RUnlock()

	if err := xlaCausalOps.errIfClosed("Counterfactual"); err != nil {
		return nil, err
	}

	if len(shape) < 2 {
		return nil, fmt.Errorf("Counterfactual: len(shape) < 2")
	}

	n, nCF := shape[0], shape[1]

	if n <= 0 || nCF <= 0 || n > math.MaxInt32 || nCF > math.MaxInt32 {
		return nil, fmt.Errorf("Counterfactual: invalid n=%d nCF=%d", n, nCF)
	}

	if int64(n)*int64(nCF) > int64(math.MaxInt) {
		return nil, fmt.Errorf("Counterfactual: N*N_cf overflow")
	}

	if len(data) < 4 {
		return nil, fmt.Errorf("Counterfactual: len(data) < 4")
	}

	if len(data[0]) < n || len(data[1]) < n || len(data[2]) < n || len(data[3]) < nCF {
		return nil, fmt.Errorf("Counterfactual: data length mismatch for n=%d nCF=%d", n, nCF)
	}

	cn, err := cIntCausal("Counterfactual.n", n)
	if err != nil {
		return nil, err
	}

	cncf, err := cIntCausal("Counterfactual.nCF", nCF)
	if err != nil {
		return nil, err
	}

	yCF := make([]float64, n*nCF)
	rc := C.xla_causal_counterfactual(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&data[3][0])),
		(*C.double)(unsafe.Pointer(&yCF[0])),
		cn, cncf,
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_causal_counterfactual failed (rc=%d)", rc)
	}

	return yCF, nil
}

/*
IVEstimate computes the 2SLS instrumental variable estimate.
shape=[T, N_z, N_x, N_y], data[0]=Z, data[1]=X, data[2]=Y
Returns beta_iv [N_x*N_y].
*/
func (xlaCausalOps *XLACausalOps) IVEstimate(shape []int, data ...[]float64) ([]float64, error) {
	xlaCausalOps.mu.RLock()
	defer xlaCausalOps.mu.RUnlock()

	if err := xlaCausalOps.errIfClosed("IVEstimate"); err != nil {
		return nil, err
	}

	if len(shape) < 4 {
		return nil, fmt.Errorf("IVEstimate: len(shape) < 4")
	}

	t, nz, nx, ny := shape[0], shape[1], shape[2], shape[3]

	if t <= 0 || nz <= 0 || nx <= 0 || ny <= 0 {
		return nil, fmt.Errorf("IVEstimate: non-positive dimension")
	}

	if t > math.MaxInt32 || nz > math.MaxInt32 || nx > math.MaxInt32 || ny > math.MaxInt32 {
		return nil, fmt.Errorf("IVEstimate: dimension out of range for C.int")
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("IVEstimate: len(data) < 3")
	}

	if len(data[0]) < t*nz || len(data[1]) < t*nx || len(data[2]) < t*ny {
		return nil, fmt.Errorf("IVEstimate: data length mismatch")
	}

	ct, err := cIntCausal("IVEstimate.t", t)
	if err != nil {
		return nil, err
	}

	cnz, err := cIntCausal("IVEstimate.nz", nz)
	if err != nil {
		return nil, err
	}

	cnx, err := cIntCausal("IVEstimate.nx", nx)
	if err != nil {
		return nil, err
	}

	cny, err := cIntCausal("IVEstimate.ny", ny)
	if err != nil {
		return nil, err
	}

	betaIV := make([]float64, nx*ny)
	rc := C.xla_causal_iv(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&betaIV[0])),
		ct, cnz, cnx, cny,
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_causal_iv failed (rc=%d)", rc)
	}

	return betaIV, nil
}

/*
CATE computes the Conditional Average Treatment Effect for each observation.
shape=[T, N_x, 1], data[0]=X, data[1]=treatment, data[2]=Y
Returns cate [T].
*/
func (xlaCausalOps *XLACausalOps) CATE(shape []int, data ...[]float64) ([]float64, error) {
	xlaCausalOps.mu.RLock()
	defer xlaCausalOps.mu.RUnlock()

	if err := xlaCausalOps.errIfClosed("CATE"); err != nil {
		return nil, err
	}

	if len(shape) < 2 {
		return nil, fmt.Errorf("CATE: len(shape) < 2")
	}

	t, nx := shape[0], shape[1]

	if t <= 0 || nx <= 0 || t > math.MaxInt32 || nx > math.MaxInt32 {
		return nil, fmt.Errorf("CATE: invalid t=%d nx=%d", t, nx)
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("CATE: len(data) < 3")
	}

	if len(data[0]) < t*nx || len(data[1]) < t || len(data[2]) < t {
		return nil, fmt.Errorf("CATE: data length mismatch for t=%d nx=%d", t, nx)
	}

	ct, err := cIntCausal("CATE.t", t)
	if err != nil {
		return nil, err
	}

	cnx, err := cIntCausal("CATE.nx", nx)
	if err != nil {
		return nil, err
	}

	cate := make([]float64, t)
	rc := C.xla_causal_cate(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&cate[0])),
		ct, cnx,
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_causal_cate failed (rc=%d)", rc)
	}

	return cate, nil
}

/*
DAGMarkovFactorization computes per-observation log probabilities under the DAG Markov factorization.
shape=[N, T], data[0]=X [T*N], data[1]=adj [N*N]
Returns log_prob [T].
*/
func (xlaCausalOps *XLACausalOps) DAGMarkovFactorization(shape []int, data ...[]float64) ([]float64, error) {
	xlaCausalOps.mu.RLock()
	defer xlaCausalOps.mu.RUnlock()

	if err := xlaCausalOps.errIfClosed("DAGMarkovFactorization"); err != nil {
		return nil, err
	}

	if len(shape) < 2 {
		return nil, fmt.Errorf("DAGMarkovFactorization: len(shape) < 2")
	}

	n, t := shape[0], shape[1]

	if n <= 0 || t <= 0 || n > math.MaxInt32 || t > math.MaxInt32 {
		return nil, fmt.Errorf("DAGMarkovFactorization: invalid n=%d t=%d", n, t)
	}

	if len(data) < 2 {
		return nil, fmt.Errorf("DAGMarkovFactorization: len(data) < 2")
	}

	if len(data[0]) < t*n || len(data[1]) < n*n {
		return nil, fmt.Errorf("DAGMarkovFactorization: data length mismatch for n=%d t=%d", n, t)
	}

	ct, err := cIntCausal("DAGMarkovFactorization.t", t)
	if err != nil {
		return nil, err
	}

	cn, err := cIntCausal("DAGMarkovFactorization.n", n)
	if err != nil {
		return nil, err
	}

	logProb := make([]float64, t)
	rc := C.xla_causal_dag_markov(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&logProb[0])),
		ct, cn,
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_causal_dag_markov failed (rc=%d)", rc)
	}

	return logProb, nil
}
