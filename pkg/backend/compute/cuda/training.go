//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "training.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
CUDATrainingOps dispatches CUDA training loss, gradient, and metric kernels.
The operation owns no device memory between calls; each method manages its own
temporary device buffers through the native CUDA wrapper.
*/
type CUDATrainingOps struct{}

/*
NewTrainingOps instantiates CUDA training operations.
CUDA context ownership stays with the process-level CUDA runtime setup used by
the native wrappers.
*/
func NewTrainingOps() *CUDATrainingOps {
	return &CUDATrainingOps{}
}

func (trainingOps *CUDATrainingOps) MSELoss(predictions, targets []float64) ([]float64, error) {
	if err := trainingInputPair("cuda_train_mse_loss", predictions, targets); err != nil {
		return nil, err
	}

	output := make([]float64, 1)
	rc := C.cuda_train_mse_loss(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.size_t(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_train_mse_loss failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *CUDATrainingOps) CrossEntropyLoss(
	logits, targets []float64,
) ([]float64, error) {
	if err := trainingInputPair("cuda_train_cross_entropy_loss", logits, targets); err != nil {
		return nil, err
	}

	output := make([]float64, 1)
	rc := C.cuda_train_cross_entropy_loss(
		(*C.double)(unsafe.Pointer(&logits[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.size_t(len(logits)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_train_cross_entropy_loss failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *CUDATrainingOps) MSEGrad(predictions, targets []float64) ([]float64, error) {
	if err := trainingInputPair("cuda_train_mse_grad", predictions, targets); err != nil {
		return nil, err
	}

	output := make([]float64, len(predictions))
	rc := C.cuda_train_mse_grad(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.size_t(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_train_mse_grad failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *CUDATrainingOps) CrossEntropyGrad(
	logits, targets []float64,
) ([]float64, error) {
	if err := trainingInputPair("cuda_train_cross_entropy_grad", logits, targets); err != nil {
		return nil, err
	}

	output := make([]float64, len(logits))
	rc := C.cuda_train_cross_entropy_grad(
		(*C.double)(unsafe.Pointer(&logits[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.size_t(len(logits)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_train_cross_entropy_grad failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *CUDATrainingOps) Accuracy(predictions, targets []float64) ([]float64, error) {
	if err := trainingInputPair("cuda_metric_accuracy", predictions, targets); err != nil {
		return nil, err
	}

	output := make([]float64, 1)
	rc := C.cuda_metric_accuracy(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.size_t(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_metric_accuracy failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *CUDATrainingOps) F1Counts(predictions, targets []float64) ([]float64, error) {
	if err := trainingInputPair("cuda_metric_f1_counts", predictions, targets); err != nil {
		return nil, err
	}

	output := make([]float64, 4)
	rc := C.cuda_metric_f1_counts(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.size_t(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_metric_f1_counts failed (rc=%d)", rc)
	}

	return output, nil
}

func trainingInputPair(name string, left, right []float64) error {
	if len(left) == 0 {
		return fmt.Errorf("%s: first input must be non-empty", name)
	}

	if len(right) == 0 {
		return fmt.Errorf("%s: second input must be non-empty", name)
	}

	if len(left) != len(right) {
		return fmt.Errorf("%s: input lengths must match, got %d and %d", name, len(left), len(right))
	}

	return nil
}
