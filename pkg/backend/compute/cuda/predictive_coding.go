//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "predictive_coding.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
CUDAPredictiveCodingOps dispatches predictive coding kernels to the GPU via CUDA.
*/
type CUDAPredictiveCodingOps struct{}

/*
NewPredictiveCodingOps creates a CUDAPredictiveCodingOps.
*/
func NewPredictiveCodingOps() *CUDAPredictiveCodingOps { return &CUDAPredictiveCodingOps{} }

// Prediction: shape=[D_out, D_in], data[0]=W, data[1]=r → dst[D_out]
func (op *CUDAPredictiveCodingOps) Prediction(shape []int, data ...[]float64) []float64 {
	dOut, dIn := shape[0], shape[1]
	dst := make([]float64, dOut)
	rc := C.cuda_pc_prediction(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(dOut), C.int(dIn),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_pc_prediction failed (rc=%d)", rc))
	}
	return dst
}

// PredictionError: shape=[N], data[0]=x, data[1]=mu_hat, data[2]=precision(optional) → [N]
func (op *CUDAPredictiveCodingOps) PredictionError(shape []int, data ...[]float64) []float64 {
	n := shape[0]
	dst := make([]float64, n)
	var precPtr *C.double
	if len(data) >= 3 {
		precPtr = (*C.double)(unsafe.Pointer(&data[2][0]))
	}
	rc := C.cuda_pc_prediction_error(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		precPtr,
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_pc_prediction_error failed (rc=%d)", rc))
	}
	return dst
}

// UpdateRepresentation: shape=[D_in, D_out], data[0]=r, data[1]=W, data[2]=eps_lower,
// data[3]=eps_self, data[4]=lr[1] → r_new[D_in]
func (op *CUDAPredictiveCodingOps) UpdateRepresentation(shape []int, data ...[]float64) []float64 {
	dIn, dOut := shape[0], shape[1]
	dst := make([]float64, dIn)
	rc := C.cuda_pc_update_representation(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&data[3][0])),
		C.double(data[4][0]),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(dOut), C.int(dIn),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_pc_update_representation failed (rc=%d)", rc))
	}
	return dst
}

// UpdateWeights: shape=[D_out, D_in], data[0]=W, data[1]=eps, data[2]=r, data[3]=lr[1] → W_new
func (op *CUDAPredictiveCodingOps) UpdateWeights(shape []int, data ...[]float64) []float64 {
	dOut, dIn := shape[0], shape[1]
	dst := make([]float64, dOut*dIn)
	rc := C.cuda_pc_update_weights(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		C.double(data[3][0]),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(dOut), C.int(dIn),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_pc_update_weights failed (rc=%d)", rc))
	}
	return dst
}
