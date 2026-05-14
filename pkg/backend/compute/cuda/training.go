//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "training.h"
import "C"

import (
	"fmt"
	"unsafe"
)

type CUDATrainingOps struct{}

func NewTrainingOps() *CUDATrainingOps {
	return &CUDATrainingOps{}
}

func (trainingOps *CUDATrainingOps) MSELoss(predictions, targets []float64) ([]float64, error) {
	output := make([]float64, 1)
	rc := C.cuda_train_mse_loss(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_train_mse_loss failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *CUDATrainingOps) CrossEntropyLoss(
	logits, targets []float64,
) ([]float64, error) {
	output := make([]float64, 1)
	rc := C.cuda_train_cross_entropy_loss(
		(*C.double)(unsafe.Pointer(&logits[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int(len(logits)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_train_cross_entropy_loss failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *CUDATrainingOps) MSEGrad(predictions, targets []float64) ([]float64, error) {
	output := make([]float64, len(predictions))
	rc := C.cuda_train_mse_grad(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_train_mse_grad failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *CUDATrainingOps) CrossEntropyGrad(
	logits, targets []float64,
) ([]float64, error) {
	output := make([]float64, len(logits))
	rc := C.cuda_train_cross_entropy_grad(
		(*C.double)(unsafe.Pointer(&logits[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int(len(logits)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_train_cross_entropy_grad failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *CUDATrainingOps) Accuracy(predictions, targets []float64) ([]float64, error) {
	output := make([]float64, 1)
	rc := C.cuda_bench_accuracy(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_bench_accuracy failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *CUDATrainingOps) F1Counts(predictions, targets []float64) ([]float64, error) {
	output := make([]float64, 3)
	rc := C.cuda_bench_f1_counts(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_bench_f1_counts failed (rc=%d)", rc)
	}

	return output, nil
}
