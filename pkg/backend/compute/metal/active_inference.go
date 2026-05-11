//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "active_inference.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
ActiveInferenceOps dispatches Active Inference kernels to the GPU via Metal,
implementing Karl Friston's Free Energy Principle operations.
metallib must be the absolute path to active_inference.metallib.
*/
type ActiveInferenceOps struct {
	metallib string
}

/*
NewActiveInferenceOps creates and initialises an ActiveInferenceOps instance.
*/
func NewActiveInferenceOps(metallib string) (*ActiveInferenceOps, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_ai_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_ai_init failed (rc=%d): check %q exists", rc, metallib)
	}

	return &ActiveInferenceOps{metallib: metallib}, nil
}

/*
FreeEnergy computes F = 0.5*sum(mu^2 + exp(ls) - ls - 1).
shape=[N], data[0]=mu, data[1]=log_sigma. Returns scalar [1].
*/
func (metalOps *ActiveInferenceOps) FreeEnergy(shape []int, data ...[]float64) ([]float64, error) {
	n := shape[0]
	mu := toFloat32(data[0])
	ls := toFloat32(data[1])
	out := make([]float32, 1)
	rc := C.metal_ai_free_energy(
		(*C.float)(unsafe.Pointer(&mu[0])),
		(*C.float)(unsafe.Pointer(&ls[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_ai_free_energy failed (rc=%d)", rc)
	}

	return toFloat64(out), nil
}

/*
BeliefUpdate performs one gradient descent step on F.
shape=[N, lr_steps], data[0]=mu, data[1]=log_sigma, data[2]=pred_err.
Returns [mu_new || log_sigma_new] of length 2N.
*/
func (metalOps *ActiveInferenceOps) BeliefUpdate(shape []int, data ...[]float64) ([]float64, error) {
	n := shape[0]
	lr := float32(shape[1]) * 1e-4
	mu := toFloat32(data[0])
	ls := toFloat32(data[1])
	pe := toFloat32(data[2])
	out := make([]float32, 2*n)
	rc := C.metal_ai_belief_update(
		(*C.float)(unsafe.Pointer(&mu[0])),
		(*C.float)(unsafe.Pointer(&ls[0])),
		(*C.float)(unsafe.Pointer(&pe[0])),
		C.float(lr),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_ai_belief_update failed (rc=%d)", rc)
	}

	return toFloat64(out), nil
}

/*
PrecisionWeight computes out[i] = err[i] * exp(log_prec[i]).
shape=[N], data[0]=err, data[1]=log_precision. Returns [N].
*/
func (metalOps *ActiveInferenceOps) PrecisionWeight(shape []int, data ...[]float64) ([]float64, error) {
	n := shape[0]
	err := toFloat32(data[0])
	lp := toFloat32(data[1])
	out := make([]float32, n)
	rc := C.metal_ai_precision_weight(
		(*C.float)(unsafe.Pointer(&err[0])),
		(*C.float)(unsafe.Pointer(&lp[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_ai_precision_weight failed (rc=%d)", rc)
	}

	return toFloat64(out), nil
}

/*
ExpectedFreeEnergy computes G[k] = -sum_i q[i,k]*ln(q[i,k]) for each outcome k.
shape=[N,K], data[0]=q_outcomes [N*K]. Returns G [K].
*/
func (metalOps *ActiveInferenceOps) ExpectedFreeEnergy(shape []int, data ...[]float64) ([]float64, error) {
	n, k := shape[0], shape[1]
	q := toFloat32(data[0])
	out := make([]float32, k)
	rc := C.metal_ai_expected_free_energy(
		(*C.float)(unsafe.Pointer(&q[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(n), C.int(k),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_ai_expected_free_energy failed (rc=%d)", rc)
	}

	return toFloat64(out), nil
}
