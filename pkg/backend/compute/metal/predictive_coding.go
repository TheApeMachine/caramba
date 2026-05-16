//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "predictive_coding.h"
import "C"

import (
	"fmt"
	"strings"
	"sync"
	"unsafe"
)

/*
MetalPredictiveCodingOps dispatches predictive coding on Metal; calls are serialised by a mutex.
*/
type MetalPredictiveCodingOps struct {
	mu       sync.Mutex
	metallib string
	runtime  *MetalRuntime
}

/*
NewPredictiveCodingOps initialises GPU pipelines from predictive_coding.metallib (see Makefile).
*/
func NewPredictiveCodingOps(metallib string) (*MetalPredictiveCodingOps, error) {
	if strings.TrimSpace(metallib) == "" {
		return nil, fmt.Errorf("NewPredictiveCodingOps: metallib path is empty")
	}

	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_pc_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_pc_init failed (rc=%d): check %q exists", rc, metallib)
	}

	runtime, err := newStandaloneMetalRuntime()
	if err != nil {
		return nil, err
	}

	return &MetalPredictiveCodingOps{metallib: metallib, runtime: runtime}, nil
}

/*
Close releases resources from metal_pc_init.
*/
func (op *MetalPredictiveCodingOps) Close() error {
	op.mu.Lock()
	defer op.mu.Unlock()

	if rc := C.metal_pc_shutdown(); rc != 0 {
		return fmt.Errorf("metal_pc_shutdown failed (rc=%d)", rc)
	}

	return nil
}

// Prediction: shape=[D_out, D_in], data[0]=W, data[1]=r → [D_out]
func (op *MetalPredictiveCodingOps) Prediction(shape []int, data ...[]float64) ([]float64, error) {
	op.mu.Lock()
	defer op.mu.Unlock()

	if len(shape) < 2 {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.Prediction: need len(shape) >= 2")
	}

	dOut, dIn := shape[0], shape[1]

	if dOut <= 0 || dIn <= 0 {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.Prediction: need positive D_out and D_in")
	}

	if len(data) < 2 {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.Prediction: need len(data) >= 2")
	}

	if len(data[0]) < dOut*dIn || len(data[1]) < dIn {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.Prediction: W needs %d elements, r needs %d", dOut*dIn, dIn)
	}

	W := toFloat32(data[0])
	r := toFloat32(data[1])
	dst := make([]float32, dOut)

	rc := C.metal_pc_prediction(
		(*C.float)(unsafe.Pointer(&W[0])),
		(*C.float)(unsafe.Pointer(&r[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(dOut), C.int(dIn),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_pc_prediction failed (rc=%d)", rc)
	}

	return toFloat64(dst), nil
}

// PredictionError: shape=[N], data[0]=x, data[1]=mu_hat; optional data[2]=prec → [N]
func (op *MetalPredictiveCodingOps) PredictionError(shape []int, data ...[]float64) ([]float64, error) {
	op.mu.Lock()
	defer op.mu.Unlock()

	if len(shape) < 1 || shape[0] <= 0 {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.PredictionError: need shape[0] > 0")
	}

	n := shape[0]

	if len(data) < 2 {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.PredictionError: need len(data) >= 2")
	}

	if len(data[0]) < n || len(data[1]) < n {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.PredictionError: x and mu_hat length >= %d", n)
	}

	x := toFloat32(data[0])
	mu := toFloat32(data[1])
	dst := make([]float32, n)

	var precPtr *C.float
	var precHolder []float32

	if len(data) >= 3 {
		precHolder = toFloat32(data[2])

		if len(precHolder) >= n {
			precPtr = (*C.float)(unsafe.Pointer(&precHolder[0]))
		}
	}

	rc := C.metal_pc_prediction_error(
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&mu[0])),
		precPtr,
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_pc_prediction_error failed (rc=%d)", rc)
	}

	return toFloat64(dst), nil
}

// UpdateRepresentation: shape=[D_in, D_out], data[0]=r, data[1]=W, data[2]=eps_lower,
// data[3]=eps_self, data[4]=lr[1] → r_new[D_in]
func (op *MetalPredictiveCodingOps) UpdateRepresentation(shape []int, data ...[]float64) ([]float64, error) {
	op.mu.Lock()
	defer op.mu.Unlock()

	if len(shape) < 2 {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.UpdateRepresentation: need len(shape) >= 2")
	}

	dIn, dOut := shape[0], shape[1]

	if dIn <= 0 || dOut <= 0 {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.UpdateRepresentation: need positive D_in and D_out")
	}

	if len(data) < 5 {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.UpdateRepresentation: need len(data) >= 5")
	}

	if len(data[0]) < dIn || len(data[1]) < dOut*dIn || len(data[2]) < dOut || len(data[3]) < dIn || len(data[4]) < 1 {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.UpdateRepresentation: slice length mismatch")
	}

	r := toFloat32(data[0])
	W := toFloat32(data[1])
	epsLower := toFloat32(data[2])
	epsSelf := toFloat32(data[3])
	lr := float32(data[4][0])
	dst := make([]float32, dIn)

	rc := C.metal_pc_update_representation(
		(*C.float)(unsafe.Pointer(&r[0])),
		(*C.float)(unsafe.Pointer(&W[0])),
		(*C.float)(unsafe.Pointer(&epsLower[0])),
		(*C.float)(unsafe.Pointer(&epsSelf[0])),
		C.float(lr),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(dOut), C.int(dIn),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_pc_update_representation failed (rc=%d)", rc)
	}

	return toFloat64(dst), nil
}

// UpdateWeights: shape=[D_out, D_in], data[0]=W, data[1]=eps, data[2]=r, data[3]=lr[1] → W_new
func (op *MetalPredictiveCodingOps) UpdateWeights(shape []int, data ...[]float64) ([]float64, error) {
	op.mu.Lock()
	defer op.mu.Unlock()

	if len(shape) < 2 {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.UpdateWeights: need len(shape) >= 2")
	}

	dOut, dIn := shape[0], shape[1]

	if dOut <= 0 || dIn <= 0 {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.UpdateWeights: need positive D_out and D_in")
	}

	if len(data) < 4 {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.UpdateWeights: need len(data) >= 4")
	}

	if len(data[3]) < 1 {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.UpdateWeights: data[3] must contain lr")
	}

	if len(data[0]) < dOut*dIn || len(data[1]) < dOut || len(data[2]) < dIn {
		return nil, fmt.Errorf("MetalPredictiveCodingOps.UpdateWeights: slice length mismatch")
	}

	W := toFloat32(data[0])
	eps := toFloat32(data[1])
	r := toFloat32(data[2])
	lr := float32(data[3][0])
	dst := make([]float32, dOut*dIn)

	rc := C.metal_pc_update_weights(
		(*C.float)(unsafe.Pointer(&W[0])),
		(*C.float)(unsafe.Pointer(&eps[0])),
		(*C.float)(unsafe.Pointer(&r[0])),
		C.float(lr),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(dOut), C.int(dIn),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_pc_update_weights failed (rc=%d)", rc)
	}

	return toFloat64(dst), nil
}
