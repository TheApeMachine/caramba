//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "xla_predictive_coding.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
XLAPredictiveCodingOps dispatches predictive coding operations to the XLA runtime via PJRT.
*/
type XLAPredictiveCodingOps struct {
	platform string
}

/*
NewPredictiveCodingOps initialises the PJRT client for the given platform.
*/
func NewPredictiveCodingOps(platform string) (*XLAPredictiveCodingOps, error) {
	config, err := NewPJRTConfig(platform)

	if err != nil {
		return nil, err
	}

	if err := config.ValidateRuntime(); err != nil {
		return nil, err
	}

	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_pc_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_pc_init failed for platform %q", platform)
	}

	return &XLAPredictiveCodingOps{platform: platform}, nil
}

/*
Shutdown releases all PJRT predictive coding resources.
*/
func (op *XLAPredictiveCodingOps) Shutdown() { C.xla_pc_shutdown() }

// Prediction: shape=[D_out, D_in], data[0]=W, data[1]=r → [D_out]
func (op *XLAPredictiveCodingOps) Prediction(shape []int, data ...[]float64) []float64 {
	dOut, dIn := shape[0], shape[1]
	dst := make([]float64, dOut)
	rc := C.xla_pc_prediction(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(dOut), C.int(dIn),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_pc_prediction failed"))
	}
	return dst
}

// PredictionError: shape=[N], data[0]=x, data[1]=mu_hat, data[2]=prec(optional) → [N]
func (op *XLAPredictiveCodingOps) PredictionError(shape []int, data ...[]float64) []float64 {
	n := shape[0]
	dst := make([]float64, n)
	usePrec := C.int(0)
	var precPtr *C.double

	if len(data) >= 3 {
		usePrec = 1
		precPtr = (*C.double)(unsafe.Pointer(&data[2][0]))
	}

	rc := C.xla_pc_prediction_error(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		precPtr,
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n), usePrec,
	)

	if rc != 0 {
		panic(fmt.Sprintf("xla_pc_prediction_error failed"))
	}

	return dst
}

// UpdateRepresentation: shape=[D_in, D_out], data[0]=r, data[1]=W, data[2]=eps_lower,
// data[3]=eps_self, data[4]=lr[1] → r_new[D_in]
func (op *XLAPredictiveCodingOps) UpdateRepresentation(shape []int, data ...[]float64) []float64 {
	dIn, dOut := shape[0], shape[1]
	dst := make([]float64, dIn)
	rc := C.xla_pc_update_representation(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&data[3][0])),
		C.double(data[4][0]),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(dOut), C.int(dIn),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_pc_update_representation failed"))
	}
	return dst
}

// UpdateWeights: shape=[D_out, D_in], data[0]=W, data[1]=eps, data[2]=r, data[3]=lr[1] → W_new
func (op *XLAPredictiveCodingOps) UpdateWeights(shape []int, data ...[]float64) []float64 {
	dOut, dIn := shape[0], shape[1]
	dst := make([]float64, dOut*dIn)
	rc := C.xla_pc_update_weights(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		C.double(data[3][0]),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(dOut), C.int(dIn),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_pc_update_weights failed"))
	}
	return dst
}
