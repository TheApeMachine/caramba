//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "xla_active_inference.h"
import "C"

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"unsafe"
)

var xlaActiveInferenceUsers atomic.Int32

/*
XLAActiveInferenceOps dispatches Active Inference operations to the XLA runtime
via PJRT, implementing Karl Friston's Free Energy Principle with JIT compilation.

Shutdown is reference-counted: the C runtime xla_ai_shutdown runs only after the
last live instance calls Shutdown.
*/
type XLAActiveInferenceOps struct {
	shutdownOnce sync.Once
}

/*
NewActiveInferenceOps initialises the PJRT client for the given platform.
*/
func NewActiveInferenceOps(platform string) (*XLAActiveInferenceOps, error) {
	config, err := newRuntimePJRTConfig(platform)

	if err != nil {
		return nil, err
	}

	cp := C.CString(config.Platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_ai_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_ai_init failed for platform %q (rc=%d)", config.Platform, rc)
	}

	xlaActiveInferenceUsers.Add(1)

	return &XLAActiveInferenceOps{}, nil
}

/*
Shutdown releases PJRT active inference resources when the last ops instance shuts down.
*/
func (xlaOps *XLAActiveInferenceOps) Shutdown() {
	xlaOps.shutdownOnce.Do(func() {
		if xlaActiveInferenceUsers.Add(-1) == 0 {
			C.xla_ai_shutdown()
		}
	})
}

func cIntAI(name string, v int) (C.int, error) {
	if v < 0 || v > math.MaxInt32 {
		return 0, fmt.Errorf("%s: value %d out of range for C.int", name, v)
	}

	return C.int(int32(v)), nil
}

/*
FreeEnergy computes F = 0.5*sum(mu^2 + exp(ls) - ls - 1).
shape=[N], data[0]=mu, data[1]=log_sigma. Returns scalar [1].
*/
func (xlaOps *XLAActiveInferenceOps) FreeEnergy(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) < 1 {
		return nil, fmt.Errorf("XLAActiveInferenceOps.FreeEnergy: len(shape) < 1")
	}

	n := shape[0]

	if n < 0 || n > math.MaxInt32 {
		return nil, fmt.Errorf("XLAActiveInferenceOps.FreeEnergy: invalid n=%d", n)
	}

	if len(data) < 2 {
		return nil, fmt.Errorf("XLAActiveInferenceOps.FreeEnergy: len(data) < 2")
	}

	if len(data[0]) < n || len(data[1]) < n {
		return nil, fmt.Errorf("XLAActiveInferenceOps.FreeEnergy: data slices shorter than n=%d", n)
	}

	cn, err := cIntAI("FreeEnergy.n", n)
	if err != nil {
		return nil, err
	}

	out := make([]float64, 1)
	rc := C.xla_ai_free_energy(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		cn,
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_ai_free_energy failed (rc=%d)", rc)
	}

	return out, nil
}

/*
BeliefUpdate performs one gradient descent step on F.
shape=[N, lr_steps], data[0]=mu, data[1]=log_sigma, data[2]=pred_err.
lr_steps must be > 0; learning rate is float64(shape[1]) * 1e-4.
Returns [mu_new || log_sigma_new] of length 2N.
*/
func (xlaOps *XLAActiveInferenceOps) BeliefUpdate(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) < 2 {
		return nil, fmt.Errorf("XLAActiveInferenceOps.BeliefUpdate: len(shape) < 2")
	}

	n := shape[0]

	if n <= 0 || n > math.MaxInt32 {
		return nil, fmt.Errorf("XLAActiveInferenceOps.BeliefUpdate: invalid n=%d", n)
	}

	if shape[1] <= 0 {
		return nil, fmt.Errorf("XLAActiveInferenceOps.BeliefUpdate: shape[1] (lr_steps) must be > 0, got %d", shape[1])
	}

	lr := float64(shape[1]) * 1e-4

	if len(data) < 3 {
		return nil, fmt.Errorf("XLAActiveInferenceOps.BeliefUpdate: len(data) < 3")
	}

	for i := 0; i < 3; i++ {
		if len(data[i]) < n {
			return nil, fmt.Errorf("XLAActiveInferenceOps.BeliefUpdate: len(data[%d])=%d, need >= %d", i, len(data[i]), n)
		}
	}

	cn, err := cIntAI("BeliefUpdate.n", n)
	if err != nil {
		return nil, err
	}

	out := make([]float64, 2*n)
	rc := C.xla_ai_belief_update(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		C.double(lr),
		(*C.double)(unsafe.Pointer(&out[0])),
		cn,
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_ai_belief_update failed (rc=%d)", rc)
	}

	return out, nil
}

/*
PrecisionWeight computes out[i] = err[i] * exp(log_prec[i]).
shape=[N], data[0]=err, data[1]=log_precision. Returns [N].
*/
func (xlaOps *XLAActiveInferenceOps) PrecisionWeight(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) < 1 {
		return nil, fmt.Errorf("XLAActiveInferenceOps.PrecisionWeight: len(shape) < 1")
	}

	n := shape[0]

	if n <= 0 || n > math.MaxInt32 {
		return nil, fmt.Errorf("XLAActiveInferenceOps.PrecisionWeight: invalid n=%d", n)
	}

	if len(data) < 2 {
		return nil, fmt.Errorf("XLAActiveInferenceOps.PrecisionWeight: len(data) < 2")
	}

	if len(data[0]) < n || len(data[1]) < n {
		return nil, fmt.Errorf("XLAActiveInferenceOps.PrecisionWeight: data slices shorter than n=%d", n)
	}

	cn, err := cIntAI("PrecisionWeight.n", n)
	if err != nil {
		return nil, err
	}

	out := make([]float64, n)
	rc := C.xla_ai_precision_weight(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		cn,
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_ai_precision_weight failed (rc=%d)", rc)
	}

	return out, nil
}

/*
ExpectedFreeEnergy computes G[k] = -sum_i q[i,k]*ln(q[i,k]+eps) for each outcome k.
shape=[N,K], data[0]=q_outcomes [N*K]; optional data[1][0]=eps (>0), default 1e-12. Returns G [K].
*/
func (xlaOps *XLAActiveInferenceOps) ExpectedFreeEnergy(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) < 2 {
		return nil, fmt.Errorf("XLAActiveInferenceOps.ExpectedFreeEnergy: len(shape) < 2")
	}

	n, k := shape[0], shape[1]

	if n <= 0 || k <= 0 || n > math.MaxInt32 || k > math.MaxInt32 {
		return nil, fmt.Errorf("XLAActiveInferenceOps.ExpectedFreeEnergy: invalid n=%d k=%d", n, k)
	}

	if len(data) < 1 || len(data[0]) < n*k {
		return nil, fmt.Errorf(
			"XLAActiveInferenceOps.ExpectedFreeEnergy: len(data[0])=%d, need n*k=%d",
			len(data[0]), n*k,
		)
	}

	eps := 1e-12
	if len(data) >= 2 && len(data[1]) >= 1 && data[1][0] > 0 {
		eps = data[1][0]
	}

	cn, err := cIntAI("ExpectedFreeEnergy.n", n)
	if err != nil {
		return nil, err
	}

	ck, err := cIntAI("ExpectedFreeEnergy.k", k)
	if err != nil {
		return nil, err
	}

	out := make([]float64, k)
	rc := C.xla_ai_expected_free_energy(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		cn, ck,
		C.double(eps),
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_ai_expected_free_energy failed (rc=%d)", rc)
	}

	return out, nil
}
