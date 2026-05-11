//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "predictive_coding.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
MetalPredictiveCodingOps dispatches predictive coding kernels via Metal.
*/
type MetalPredictiveCodingOps struct {
	metallib string
}

/*
NewPredictiveCodingOps creates and initialises a MetalPredictiveCodingOps instance.
*/
func NewPredictiveCodingOps(metallib string) (*MetalPredictiveCodingOps, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_pc_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_pc_init failed (rc=%d): check %q exists", rc, metallib)
	}

	return &MetalPredictiveCodingOps{metallib: metallib}, nil
}

// Prediction: shape=[D_out, D_in], data[0]=W, data[1]=r → [D_out]
func (op *MetalPredictiveCodingOps) Prediction(shape []int, data ...[]float64) ([]float64, error) {
	dOut, dIn := shape[0], shape[1]
	W, r := toFloat32(data[0]), toFloat32(data[1])
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

// PredictionError: shape=[N], data[0]=x, data[1]=mu_hat, data[2]=prec(optional) → [N]
func (op *MetalPredictiveCodingOps) PredictionError(shape []int, data ...[]float64) ([]float64, error) {
	n := shape[0]
	x, mu := toFloat32(data[0]), toFloat32(data[1])
	dst := make([]float32, n)

	var precPtr *C.float
	usePrec := C.int(0)

	if len(data) >= 3 {
		prec := toFloat32(data[2])
		precPtr = (*C.float)(unsafe.Pointer(&prec[0]))
		usePrec = 1
	}

	rc := C.metal_pc_prediction_error(
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&mu[0])),
		precPtr,
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n), usePrec,
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_pc_prediction_error failed (rc=%d)", rc)
	}

	return toFloat64(dst), nil
}

// UpdateRepresentation: shape=[D_in, D_out], data[0]=r, data[1]=W, data[2]=eps_lower,
// data[3]=eps_self, data[4]=lr[1] → r_new[D_in]
func (op *MetalPredictiveCodingOps) UpdateRepresentation(shape []int, data ...[]float64) ([]float64, error) {
	dIn, dOut := shape[0], shape[1]
	r, W := toFloat32(data[0]), toFloat32(data[1])
	epsLower, epsSelf := toFloat32(data[2]), toFloat32(data[3])
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
	dOut, dIn := shape[0], shape[1]
	W, eps, r := toFloat32(data[0]), toFloat32(data[1]), toFloat32(data[2])
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
