//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "active_inference.h"
import "C"

import (
	"fmt"
	"strings"
	"sync"
	"unsafe"
)

// DefaultExpectedFreeEnergyEps is passed to C.metal_ai_expected_free_energy for log stability.
const DefaultExpectedFreeEnergyEps = float32(1e-12)

/*
ActiveInferenceOps dispatches Active Inference kernels.
Inputs are float64 on the Go side; the reference Metal bridge computes in float32.

Safe for concurrent use from multiple goroutines: a mutex serialises calls on this wrapper.

metallib must be an absolute path to active_inference.metallib (see repo Makefile).
*/
type ActiveInferenceOps struct {
	mu       sync.Mutex
	metallib string
	runtime  *MetalRuntime
}

/*
NewActiveInferenceOps creates and initialises an ActiveInferenceOps instance.
*/
func NewActiveInferenceOps(metallib string) (*ActiveInferenceOps, error) {
	if strings.TrimSpace(metallib) == "" {
		return nil, fmt.Errorf("NewActiveInferenceOps: metallib path is empty")
	}

	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_ai_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_ai_init failed (rc=%d): check %q exists", rc, metallib)
	}

	runtime, err := newStandaloneMetalRuntime()
	if err != nil {
		return nil, err
	}

	return &ActiveInferenceOps{metallib: metallib, runtime: runtime}, nil
}

/*
Close releases resources acquired by metal_ai_init. Idempotent with respect to the C layer.
*/
func (metalOps *ActiveInferenceOps) Close() error {
	metalOps.mu.Lock()
	defer metalOps.mu.Unlock()

	if rc := C.metal_ai_cleanup(); rc != 0 {
		return fmt.Errorf("metal_ai_cleanup failed (rc=%d)", rc)
	}

	return nil
}

/*
FreeEnergy computes F = 0.5*sum(mu^2 + exp(ls) - ls - 1).
shape=[N], data[0]=mu, data[1]=log_sigma. Returns scalar [1].
*/
func (metalOps *ActiveInferenceOps) FreeEnergy(shape []int, data ...[]float64) ([]float64, error) {
	metalOps.mu.Lock()
	defer metalOps.mu.Unlock()

	if len(shape) < 1 {
		return nil, fmt.Errorf("ActiveInferenceOps.FreeEnergy: need len(shape) >= 1")
	}

	n := shape[0]

	if n < 0 {
		return nil, fmt.Errorf("ActiveInferenceOps.FreeEnergy: invalid n=%d", n)
	}

	if n == 0 {
		out := make([]float32, 1)

		rc := C.metal_ai_free_energy(nil, nil, (*C.float)(unsafe.Pointer(&out[0])), C.int(0))
		if rc != 0 {
			return nil, fmt.Errorf("metal_ai_free_energy failed (rc=%d)", rc)
		}

		return toFloat64(out), nil
	}

	if len(data) < 2 {
		return nil, fmt.Errorf("ActiveInferenceOps.FreeEnergy: need len(data) >= 2")
	}

	if len(data[0]) != n || len(data[1]) != n {
		return nil, fmt.Errorf("ActiveInferenceOps.FreeEnergy: need len(data[0])==len(data[1])==%d", n)
	}

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
shape=[N, raw_lr_steps]: effective learning rate is float32(raw_lr_steps) * 1e-4 (same convention as the CPU op).
data[0]=mu, data[1]=log_sigma, data[2]=pred_err.
Returns [mu_new || log_sigma_new] of length 2N.
*/
func (metalOps *ActiveInferenceOps) BeliefUpdate(shape []int, data ...[]float64) ([]float64, error) {
	metalOps.mu.Lock()
	defer metalOps.mu.Unlock()

	if len(shape) < 2 {
		return nil, fmt.Errorf("ActiveInferenceOps.BeliefUpdate: need len(shape) >= 2")
	}

	n := shape[0]

	if n <= 0 {
		return nil, fmt.Errorf("ActiveInferenceOps.BeliefUpdate: need shape[0] > 0")
	}

	rawSteps := shape[1]

	if rawSteps <= 0 {
		return nil, fmt.Errorf("ActiveInferenceOps.BeliefUpdate: shape[1] (raw lr steps) must be > 0")
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("ActiveInferenceOps.BeliefUpdate: need len(data) >= 3")
	}

	if len(data[0]) != n || len(data[1]) != n || len(data[2]) != n {
		return nil, fmt.Errorf("ActiveInferenceOps.BeliefUpdate: each data slice must have length %d", n)
	}

	lr := float32(rawSteps) * 1e-4
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
	metalOps.mu.Lock()
	defer metalOps.mu.Unlock()

	if len(shape) < 1 || shape[0] <= 0 {
		return nil, fmt.Errorf("ActiveInferenceOps.PrecisionWeight: need shape[0] > 0")
	}

	n := shape[0]

	if len(data) < 2 {
		return nil, fmt.Errorf("ActiveInferenceOps.PrecisionWeight: need len(data) >= 2")
	}

	if len(data[0]) < n || len(data[1]) < n {
		return nil, fmt.Errorf("ActiveInferenceOps.PrecisionWeight: need len(err) and len(log_precision) >= %d", n)
	}

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
ExpectedFreeEnergy computes G[k] = -sum_i clamp(q[i,k],0,1)*ln(clamp(q[i,k],0,1)+eps).
shape=[N,K], data[0]=q_outcomes [N*K] row-major. Returns G [K].
eps is DefaultExpectedFreeEnergyEps (1e-12).
*/
func (metalOps *ActiveInferenceOps) ExpectedFreeEnergy(shape []int, data ...[]float64) ([]float64, error) {
	metalOps.mu.Lock()
	defer metalOps.mu.Unlock()

	if len(shape) < 2 {
		return nil, fmt.Errorf("ActiveInferenceOps.ExpectedFreeEnergy: need len(shape) >= 2")
	}

	n, k := shape[0], shape[1]

	if n <= 0 || k <= 0 {
		return nil, fmt.Errorf("ActiveInferenceOps.ExpectedFreeEnergy: need n>0 and k>0")
	}

	if len(data) < 1 {
		return nil, fmt.Errorf("ActiveInferenceOps.ExpectedFreeEnergy: missing q_outcomes")
	}

	if len(data[0]) != n*k {
		return nil, fmt.Errorf("ActiveInferenceOps.ExpectedFreeEnergy: len(q)=%d need %d*%d", len(data[0]), n, k)
	}

	eps := DefaultExpectedFreeEnergyEps
	q := toFloat32(data[0])
	out := make([]float32, k)

	rc := C.metal_ai_expected_free_energy(
		(*C.float)(unsafe.Pointer(&q[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(n), C.int(k),
		C.float(eps),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_ai_expected_free_energy failed (rc=%d)", rc)
	}

	return toFloat64(out), nil
}
