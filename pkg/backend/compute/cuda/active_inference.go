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

func activeInferencePanic(method, detail string) {
	panic("CUDAActiveInferenceOps." + method + ": " + detail)
}

func activeInferenceShapeLen(method string, shape []int, need int) {
	if len(shape) < need {
		activeInferencePanic(method, fmt.Sprintf("len(shape)=%d, need >= %d", len(shape), need))
	}
}

func activeInferenceDataLen(method string, data [][]float64, need int) {
	if len(data) < need {
		activeInferencePanic(method, fmt.Sprintf("len(data)=%d, need >= %d", len(data), need))
	}
}

func activeInferenceSliceAtLeast(method, name string, slice []float64, minLen int) {
	if slice == nil {
		activeInferencePanic(method, name+" is nil")
	}

	if len(slice) < minLen {
		activeInferencePanic(method, fmt.Sprintf("len(%s)=%d, need >= %d", name, len(slice), minLen))
	}
}

/*
FreeEnergy computes F = 0.5*sum(mu^2 + exp(ls) - ls - 1).
shape=[N], data[0]=mu, data[1]=log_sigma. Returns scalar [1].
*/
func (cudaOps *CUDAActiveInferenceOps) FreeEnergy(shape []int, data ...[]float64) []float64 {
	activeInferenceShapeLen("FreeEnergy", shape, 1)
	n := shape[0]

	if n < 0 {
		activeInferencePanic("FreeEnergy", fmt.Sprintf("shape[0] (n) must be non-negative, got n=%d", n))
	}

	activeInferenceDataLen("FreeEnergy", data, 2)

	if n == 0 {
		out := make([]float64, 1)
		rc := C.cuda_ai_free_energy(nil, nil, (*C.double)(unsafe.Pointer(&out[0])), 0)

		if rc != 0 {
			panic(fmt.Sprintf("cuda_ai_free_energy failed (rc=%d)", rc))
		}

		return out
	}

	activeInferenceSliceAtLeast("FreeEnergy", "data[0] (mu)", data[0], n)
	activeInferenceSliceAtLeast("FreeEnergy", "data[1] (log_sigma)", data[1], n)

	out := make([]float64, 1)
	rc := C.cuda_ai_free_energy(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		panic(fmt.Sprintf("cuda_ai_free_energy failed (rc=%d)", rc))
	}

	return out
}

/*
BeliefUpdate performs one gradient descent step on F.
shape=[N, lr_steps], data[0]=mu, data[1]=log_sigma, data[2]=pred_err.
Returns [mu_new || log_sigma_new] of length 2N.
*/
func (cudaOps *CUDAActiveInferenceOps) BeliefUpdate(shape []int, data ...[]float64) []float64 {
	activeInferenceShapeLen("BeliefUpdate", shape, 2)
	n := shape[0]

	if n <= 0 {
		activeInferencePanic("BeliefUpdate", fmt.Sprintf("shape[0] (n) must be positive, got n=%d", n))
	}

	activeInferenceDataLen("BeliefUpdate", data, 3)
	activeInferenceSliceAtLeast("BeliefUpdate", "data[0] (mu)", data[0], n)
	activeInferenceSliceAtLeast("BeliefUpdate", "data[1] (log_sigma)", data[1], n)
	activeInferenceSliceAtLeast("BeliefUpdate", "data[2] (pred_err)", data[2], n)

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
		panic(fmt.Sprintf("cuda_ai_belief_update failed (rc=%d)", rc))
	}

	return out
}

/*
PrecisionWeight computes out[i] = err[i] * exp(log_prec[i]).
shape=[N], data[0]=err, data[1]=log_precision. Returns [N].
*/
func (cudaOps *CUDAActiveInferenceOps) PrecisionWeight(shape []int, data ...[]float64) []float64 {
	activeInferenceShapeLen("PrecisionWeight", shape, 1)
	n := shape[0]

	if n < 0 {
		activeInferencePanic("PrecisionWeight", fmt.Sprintf("shape[0] (n) must be non-negative, got n=%d", n))
	}

	activeInferenceDataLen("PrecisionWeight", data, 2)
	if n == 0 {
		return []float64{}
	}

	activeInferenceSliceAtLeast("PrecisionWeight", "data[0] (err)", data[0], n)
	activeInferenceSliceAtLeast("PrecisionWeight", "data[1] (log_prec)", data[1], n)

	out := make([]float64, n)
	rc := C.cuda_ai_precision_weight(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		panic(fmt.Sprintf("cuda_ai_precision_weight failed (rc=%d)", rc))
	}

	return out
}

/*
ExpectedFreeEnergy computes G[k] = -sum_i q[i,k]*ln(q[i,k]) for each outcome k.
shape=[N,K], data[0]=q_outcomes [N*K]. Returns G [K].
*/
func (cudaOps *CUDAActiveInferenceOps) ExpectedFreeEnergy(shape []int, data ...[]float64) []float64 {
	activeInferenceShapeLen("ExpectedFreeEnergy", shape, 2)
	n, k := shape[0], shape[1]

	if n <= 0 || k <= 0 {
		activeInferencePanic("ExpectedFreeEnergy", fmt.Sprintf("need n>0 and k>0, got n=%d k=%d", n, k))
	}

	activeInferenceDataLen("ExpectedFreeEnergy", data, 1)

	need := int64(n) * int64(k)
	maxInt := int64(int(^uint(0) >> 1))

	if need > maxInt {
		activeInferencePanic("ExpectedFreeEnergy", fmt.Sprintf("n*k overflows int (n=%d k=%d)", n, k))
	}

	needInt := int(need)
	activeInferenceSliceAtLeast("ExpectedFreeEnergy", "data[0] (q_outcomes)", data[0], needInt)

	out := make([]float64, k)
	rc := C.cuda_ai_expected_free_energy(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n), C.int(k),
	)

	if rc != 0 {
		panic(fmt.Sprintf("cuda_ai_expected_free_energy failed (rc=%d)", rc))
	}

	return out
}
