//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "xla_predictive_coding.h"
import "C"

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

/*
XLAPredictiveCodingOps dispatches predictive coding operations to the XLA runtime via PJRT.
*/
type XLAPredictiveCodingOps struct {
	pjrtConfig PJRTConfig
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

	cp := C.CString(config.Platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_pc_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_pc_init failed for platform %q (rc=%d)", config.Platform, rc)
	}

	return &XLAPredictiveCodingOps{pjrtConfig: config}, nil
}

/*
Shutdown releases all PJRT predictive coding resources.
*/
func (op *XLAPredictiveCodingOps) Shutdown() { C.xla_pc_shutdown() }

// RuntimeConfig returns the validated PJRT layout from construction.
func (op *XLAPredictiveCodingOps) RuntimeConfig() PJRTConfig {
	return op.pjrtConfig
}

func cIntPC(name string, v int) (C.int, error) {
	if v < 0 || v > math.MaxInt32 {
		return 0, fmt.Errorf("%s: value %d out of range for C.int", name, v)
	}

	return C.int(int32(v)), nil
}

// Prediction: shape=[D_out, D_in], data[0]=W, data[1]=r → [D_out]
func (op *XLAPredictiveCodingOps) Prediction(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) < 2 {
		return nil, fmt.Errorf("Prediction: len(shape) < 2")
	}

	dOut, dIn := shape[0], shape[1]

	if dOut <= 0 || dIn <= 0 || dOut > math.MaxInt32 || dIn > math.MaxInt32 {
		return nil, fmt.Errorf("Prediction: invalid dOut=%d dIn=%d", dOut, dIn)
	}

	if int64(dOut)*int64(dIn) > int64(math.MaxInt) {
		return nil, fmt.Errorf("Prediction: dOut*dIn overflow")
	}

	if len(data) < 2 {
		return nil, fmt.Errorf("Prediction: len(data) < 2")
	}

	if len(data[0]) < dOut*dIn || len(data[1]) < dIn {
		return nil, fmt.Errorf("Prediction: data slice length mismatch")
	}

	cdOut, err := cIntPC("Prediction.dOut", dOut)
	if err != nil {
		return nil, err
	}

	cdIn, err := cIntPC("Prediction.dIn", dIn)
	if err != nil {
		return nil, err
	}

	dst := make([]float64, dOut)
	rc := C.xla_pc_prediction(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		cdOut, cdIn,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(dst)

	if rc != 0 {
		return nil, fmt.Errorf("xla_pc_prediction failed (rc=%d)", rc)
	}

	return dst, nil
}

// PredictionError: shape=[N], data[0]=x, data[1]=mu_hat, data[2]=prec(optional) → [N]
func (op *XLAPredictiveCodingOps) PredictionError(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) < 1 {
		return nil, fmt.Errorf("PredictionError: len(shape) < 1")
	}

	n := shape[0]

	if n <= 0 || n > math.MaxInt32 {
		return nil, fmt.Errorf("PredictionError: invalid n=%d", n)
	}

	if len(data) < 2 {
		return nil, fmt.Errorf("PredictionError: len(data) < 2")
	}

	if len(data[0]) < n || len(data[1]) < n {
		return nil, fmt.Errorf("PredictionError: data[0]/data[1] shorter than n=%d", n)
	}

	usePrec := C.int(0)

	var precPtr *C.double

	if len(data) >= 3 {
		usePrec = 1

		if len(data[2]) < n {
			return nil, fmt.Errorf("PredictionError: len(data[2])=%d, need >= %d when precision enabled", len(data[2]), n)
		}

		precPtr = (*C.double)(unsafe.Pointer(&data[2][0]))
	}

	cn, err := cIntPC("PredictionError.n", n)
	if err != nil {
		return nil, err
	}

	dst := make([]float64, n)
	rc := C.xla_pc_prediction_error(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		precPtr,
		(*C.double)(unsafe.Pointer(&dst[0])),
		cn, usePrec,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(dst)

	if rc != 0 {
		return nil, fmt.Errorf("xla_pc_prediction_error failed (rc=%d)", rc)
	}

	return dst, nil
}

// UpdateRepresentation: shape=[D_in, D_out], data[0]=r, data[1]=W, data[2]=eps_lower,
// data[3]=eps_self, data[4]=lr[1] → r_new[D_in]
func (op *XLAPredictiveCodingOps) UpdateRepresentation(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) < 2 {
		return nil, fmt.Errorf("UpdateRepresentation: len(shape) < 2")
	}

	dIn, dOut := shape[0], shape[1]

	if dIn <= 0 || dOut <= 0 || dIn > math.MaxInt32 || dOut > math.MaxInt32 {
		return nil, fmt.Errorf("UpdateRepresentation: invalid dIn=%d dOut=%d", dIn, dOut)
	}

	if int64(dOut)*int64(dIn) > int64(math.MaxInt) {
		return nil, fmt.Errorf("UpdateRepresentation: dOut*dIn overflow")
	}

	if len(data) < 5 {
		return nil, fmt.Errorf("UpdateRepresentation: len(data) < 5")
	}

	wLen := dOut * dIn

	if len(data[0]) < dIn || len(data[1]) < wLen || len(data[2]) < dOut || len(data[3]) < dOut || len(data[4]) < 1 {
		return nil, fmt.Errorf("UpdateRepresentation: data slice length mismatch")
	}

	cdOut, err := cIntPC("UpdateRepresentation.dOut", dOut)
	if err != nil {
		return nil, err
	}

	cdIn, err := cIntPC("UpdateRepresentation.dIn", dIn)
	if err != nil {
		return nil, err
	}

	dst := make([]float64, dIn)
	rc := C.xla_pc_update_representation(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&data[3][0])),
		C.double(data[4][0]),
		(*C.double)(unsafe.Pointer(&dst[0])),
		cdOut, cdIn,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(dst)

	if rc != 0 {
		return nil, fmt.Errorf("xla_pc_update_representation failed (rc=%d)", rc)
	}

	return dst, nil
}

// UpdateWeights: shape=[D_out, D_in], data[0]=W, data[1]=eps, data[2]=r, data[3]=lr[1] → W_new
func (op *XLAPredictiveCodingOps) UpdateWeights(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) < 2 {
		return nil, fmt.Errorf("UpdateWeights: len(shape) < 2")
	}

	dOut, dIn := shape[0], shape[1]

	if dOut <= 0 || dIn <= 0 || dOut > math.MaxInt32 || dIn > math.MaxInt32 {
		return nil, fmt.Errorf("UpdateWeights: invalid dOut=%d dIn=%d", dOut, dIn)
	}

	wLen := dOut * dIn

	if int64(wLen) > int64(math.MaxInt) {
		return nil, fmt.Errorf("UpdateWeights: dOut*dIn overflow")
	}

	if len(data) < 4 {
		return nil, fmt.Errorf("UpdateWeights: len(data) < 4")
	}

	if len(data[0]) < wLen || len(data[1]) < dOut || len(data[2]) < dIn || len(data[3]) < 1 {
		return nil, fmt.Errorf("UpdateWeights: data slice length mismatch for dOut=%d dIn=%d", dOut, dIn)
	}

	cdOut, err := cIntPC("UpdateWeights.dOut", dOut)
	if err != nil {
		return nil, err
	}

	cdIn, err := cIntPC("UpdateWeights.dIn", dIn)
	if err != nil {
		return nil, err
	}

	dst := make([]float64, wLen)
	rc := C.xla_pc_update_weights(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		C.double(data[3][0]),
		(*C.double)(unsafe.Pointer(&dst[0])),
		cdOut, cdIn,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(dst)

	if rc != 0 {
		return nil, fmt.Errorf("xla_pc_update_weights failed (rc=%d)", rc)
	}

	return dst, nil
}
