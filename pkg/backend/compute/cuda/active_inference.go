//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "active_inference.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
CUDAActiveInferenceOps dispatches Active Inference kernels to the GPU via CUDA,
implementing Karl Friston's Free Energy Principle operations.
*/
type CUDAActiveInferenceOps struct{}

/*
NewActiveInferenceOps creates a CUDAActiveInferenceOps.
*/
func NewActiveInferenceOps() *CUDAActiveInferenceOps { return &CUDAActiveInferenceOps{} }

func activeInferenceShapeLen(method string, shape []int, need int) error {
	if len(shape) < need {
		return fmt.Errorf(
			"CUDAActiveInferenceOps.%s: len(shape)=%d, need >= %d",
			method, len(shape), need,
		)
	}

	return nil
}

func activeInferenceDataLen(method string, data [][]float64, need int) error {
	if len(data) < need {
		return fmt.Errorf(
			"CUDAActiveInferenceOps.%s: len(data)=%d, need >= %d",
			method, len(data), need,
		)
	}

	return nil
}

func activeInferenceSliceAtLeast(method, name string, slice []float64, minLen int) error {
	if slice == nil {
		return fmt.Errorf("CUDAActiveInferenceOps.%s: %s is nil", method, name)
	}

	if len(slice) < minLen {
		return fmt.Errorf(
			"CUDAActiveInferenceOps.%s: len(%s)=%d, need >= %d",
			method, name, len(slice), minLen,
		)
	}

	return nil
}

/*
FreeEnergy computes F = 0.5*sum(mu^2 + exp(ls) - ls - 1).
shape=[N], data[0]=mu, data[1]=log_sigma. Returns scalar [1].
*/
func (cudaOps *CUDAActiveInferenceOps) FreeEnergy(shape []int, data ...[]float64) ([]float64, error) {
	if err := activeInferenceShapeLen("FreeEnergy", shape, 1); err != nil {
		return nil, err
	}

	n := shape[0]

	if n < 0 {
		return nil, fmt.Errorf(
			"CUDAActiveInferenceOps.FreeEnergy: shape[0] (n) must be non-negative, got n=%d",
			n,
		)
	}

	if err := activeInferenceDataLen("FreeEnergy", data, 2); err != nil {
		return nil, err
	}

	if n == 0 {
		out := make([]float64, 1)
		rc := C.cuda_ai_free_energy(nil, nil, (*C.double)(unsafe.Pointer(&out[0])), 0)

		if rc != 0 {
			return nil, fmt.Errorf("cuda_ai_free_energy failed (rc=%d)", rc)
		}

		return out, nil
	}

	if err := activeInferenceSliceAtLeast("FreeEnergy", "data[0] (mu)", data[0], n); err != nil {
		return nil, err
	}

	if err := activeInferenceSliceAtLeast(
		"FreeEnergy", "data[1] (log_sigma)", data[1], n,
	); err != nil {
		return nil, err
	}

	out := make([]float64, 1)
	rc := C.cuda_ai_free_energy(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_ai_free_energy failed (rc=%d)", rc)
	}

	return out, nil
}

/*
BeliefUpdate performs one gradient descent step on F.
shape=[N, lr_steps], data[0]=mu, data[1]=log_sigma, data[2]=pred_err.
Returns [mu_new || log_sigma_new] of length 2N.
*/
func (cudaOps *CUDAActiveInferenceOps) BeliefUpdate(shape []int, data ...[]float64) ([]float64, error) {
	if err := activeInferenceShapeLen("BeliefUpdate", shape, 2); err != nil {
		return nil, err
	}

	n := shape[0]

	if n <= 0 {
		return nil, fmt.Errorf(
			"CUDAActiveInferenceOps.BeliefUpdate: shape[0] (n) must be positive, got n=%d",
			n,
		)
	}

	if err := activeInferenceDataLen("BeliefUpdate", data, 3); err != nil {
		return nil, err
	}

	if err := activeInferenceSliceAtLeast("BeliefUpdate", "data[0] (mu)", data[0], n); err != nil {
		return nil, err
	}

	if err := activeInferenceSliceAtLeast(
		"BeliefUpdate", "data[1] (log_sigma)", data[1], n,
	); err != nil {
		return nil, err
	}

	if err := activeInferenceSliceAtLeast(
		"BeliefUpdate", "data[2] (pred_err)", data[2], n,
	); err != nil {
		return nil, err
	}

	lr := float64(shape[1]) * 1e-4
	out := make([]float64, 2*n)
	rc := C.cuda_ai_belief_update(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		C.double(lr),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_ai_belief_update failed (rc=%d)", rc)
	}

	return out, nil
}

/*
PrecisionWeight computes out[i] = err[i] * exp(log_prec[i]).
shape=[N], data[0]=err, data[1]=log_precision. Returns [N].
*/
func (cudaOps *CUDAActiveInferenceOps) PrecisionWeight(
	shape []int, data ...[]float64,
) ([]float64, error) {
	if err := activeInferenceShapeLen("PrecisionWeight", shape, 1); err != nil {
		return nil, err
	}

	n := shape[0]

	if n < 0 {
		return nil, fmt.Errorf(
			"CUDAActiveInferenceOps.PrecisionWeight: shape[0] (n) must be non-negative, got n=%d",
			n,
		)
	}

	if err := activeInferenceDataLen("PrecisionWeight", data, 2); err != nil {
		return nil, err
	}

	if n == 0 {
		return []float64{}, nil
	}

	if err := activeInferenceSliceAtLeast("PrecisionWeight", "data[0] (err)", data[0], n); err != nil {
		return nil, err
	}

	if err := activeInferenceSliceAtLeast(
		"PrecisionWeight", "data[1] (log_prec)", data[1], n,
	); err != nil {
		return nil, err
	}

	out := make([]float64, n)
	rc := C.cuda_ai_precision_weight(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_ai_precision_weight failed (rc=%d)", rc)
	}

	return out, nil
}

/*
ExpectedFreeEnergy computes G[k] = -sum_i q[i,k]*ln(q[i,k]) for each outcome k.
shape=[N,K], data[0]=q_outcomes [N*K]. Returns G [K].
*/
func (cudaOps *CUDAActiveInferenceOps) ExpectedFreeEnergy(
	shape []int, data ...[]float64,
) ([]float64, error) {
	if err := activeInferenceShapeLen("ExpectedFreeEnergy", shape, 2); err != nil {
		return nil, err
	}

	n, k := shape[0], shape[1]

	if n <= 0 || k <= 0 {
		return nil, fmt.Errorf(
			"CUDAActiveInferenceOps.ExpectedFreeEnergy: need n>0 and k>0, got n=%d k=%d",
			n, k,
		)
	}

	if err := activeInferenceDataLen("ExpectedFreeEnergy", data, 1); err != nil {
		return nil, err
	}

	need := int64(n) * int64(k)
	maxInt := int64(int(^uint(0) >> 1))

	if need > maxInt {
		return nil, fmt.Errorf(
			"CUDAActiveInferenceOps.ExpectedFreeEnergy: n*k overflows int (n=%d k=%d)",
			n, k,
		)
	}

	needInt := int(need)
	if err := activeInferenceSliceAtLeast(
		"ExpectedFreeEnergy", "data[0] (q_outcomes)", data[0], needInt,
	); err != nil {
		return nil, err
	}

	out := make([]float64, k)
	rc := C.cuda_ai_expected_free_energy(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n), C.int(k),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_ai_expected_free_energy failed (rc=%d)", rc)
	}

	return out, nil
}
