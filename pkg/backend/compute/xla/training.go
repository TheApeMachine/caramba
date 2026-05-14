//go:build cgo && xla

package xla

// #include "xla_math.h"
import "C"

import (
	"fmt"
	"unsafe"
)

type XLATrainingOps struct {
	mathOps *XLAMathOps
}

func NewTrainingOps(platform string) (*XLATrainingOps, error) {
	mathOps, err := NewMathOps(platform)

	if err != nil {
		return nil, err
	}

	return &XLATrainingOps{mathOps: mathOps}, nil
}

func (trainingOps *XLATrainingOps) Shutdown() {
	trainingOps.mathOps.Shutdown()
}

func (trainingOps *XLATrainingOps) MSELoss(predictions, targets []float64) ([]float64, error) {
	output := make([]float64, 1)
	rc := C.xla_train_mse_loss(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_train_mse_loss failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *XLATrainingOps) CrossEntropyLoss(
	logits, targets []float64,
) ([]float64, error) {
	output := make([]float64, 1)
	rc := C.xla_train_cross_entropy_loss(
		(*C.double)(unsafe.Pointer(&logits[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int(len(logits)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_train_cross_entropy_loss failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *XLATrainingOps) MSEGrad(predictions, targets []float64) ([]float64, error) {
	output := make([]float64, len(predictions))
	rc := C.xla_train_mse_grad(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_train_mse_grad failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *XLATrainingOps) CrossEntropyGrad(
	logits, targets []float64,
) ([]float64, error) {
	output := make([]float64, len(logits))
	rc := C.xla_train_cross_entropy_grad(
		(*C.double)(unsafe.Pointer(&logits[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int(len(logits)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_train_cross_entropy_grad failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *XLATrainingOps) Accuracy(predictions, targets []float64) ([]float64, error) {
	output := make([]float64, 1)
	rc := C.xla_bench_accuracy(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_bench_accuracy failed (rc=%d)", rc)
	}

	return output, nil
}

func (trainingOps *XLATrainingOps) F1Counts(predictions, targets []float64) ([]float64, error) {
	output := make([]float64, 3)
	rc := C.xla_bench_f1_counts(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_bench_f1_counts failed (rc=%d)", rc)
	}

	return output, nil
}
