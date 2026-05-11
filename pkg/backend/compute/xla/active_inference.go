//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "xla_active_inference.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
XLAActiveInferenceOps dispatches Active Inference operations to the XLA runtime
via PJRT, implementing Karl Friston's Free Energy Principle with JIT compilation.
*/
type XLAActiveInferenceOps struct {
	platform string
}

/*
NewActiveInferenceOps initialises the PJRT client for the given platform.
*/
func NewActiveInferenceOps(platform string) (*XLAActiveInferenceOps, error) {
	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_ai_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_ai_init failed for platform %q", platform)
	}

	return &XLAActiveInferenceOps{platform: platform}, nil
}

/*
Shutdown releases all PJRT active inference resources.
*/
func (xlaOps *XLAActiveInferenceOps) Shutdown() { C.xla_ai_shutdown() }

/*
FreeEnergy computes F = 0.5*sum(mu^2 + exp(ls) - ls - 1).
shape=[N], data[0]=mu, data[1]=log_sigma. Returns scalar [1].
*/
func (xlaOps *XLAActiveInferenceOps) FreeEnergy(shape []int, data ...[]float64) []float64 {
	n := shape[0]
	out := make([]float64, 1)
	rc := C.xla_ai_free_energy(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		panic(fmt.Sprintf("xla_ai_free_energy failed"))
	}

	return out
}

/*
BeliefUpdate performs one gradient descent step on F.
shape=[N, lr_steps], data[0]=mu, data[1]=log_sigma, data[2]=pred_err.
Returns [mu_new || log_sigma_new] of length 2N.
*/
func (xlaOps *XLAActiveInferenceOps) BeliefUpdate(shape []int, data ...[]float64) []float64 {
	n := shape[0]
	lr := float64(shape[1]) * 1e-4
	out := make([]float64, 2*n)
	rc := C.xla_ai_belief_update(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		C.double(lr),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		panic(fmt.Sprintf("xla_ai_belief_update failed"))
	}

	return out
}

/*
PrecisionWeight computes out[i] = err[i] * exp(log_prec[i]).
shape=[N], data[0]=err, data[1]=log_precision. Returns [N].
*/
func (xlaOps *XLAActiveInferenceOps) PrecisionWeight(shape []int, data ...[]float64) []float64 {
	n := shape[0]
	out := make([]float64, n)
	rc := C.xla_ai_precision_weight(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		panic(fmt.Sprintf("xla_ai_precision_weight failed"))
	}

	return out
}

/*
ExpectedFreeEnergy computes G[k] = -sum_i q[i,k]*ln(q[i,k]) for each outcome k.
shape=[N,K], data[0]=q_outcomes [N*K]. Returns G [K].
*/
func (xlaOps *XLAActiveInferenceOps) ExpectedFreeEnergy(shape []int, data ...[]float64) []float64 {
	n, k := shape[0], shape[1]
	out := make([]float64, k)
	rc := C.xla_ai_expected_free_energy(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n), C.int(k),
	)

	if rc != 0 {
		panic(fmt.Sprintf("xla_ai_expected_free_energy failed"))
	}

	return out
}
