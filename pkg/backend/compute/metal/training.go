//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "metal_kernel_math.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
MetalTrainingOps dispatches training losses, gradients, and benchmark kernels through
math.metallib. The kernels live beside MathOps because they share the same primitive
reductions and elementwise math pipeline.
*/
type MetalTrainingOps struct {
	metallib string
}

/*
NewTrainingOps creates and initializes MetalTrainingOps.
*/
func NewTrainingOps(metallib string) (*MetalTrainingOps, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_math_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_math_init failed (rc=%d): check %q exists", rc, metallib)
	}

	return &MetalTrainingOps{metallib: metallib}, nil
}

/*
MSELoss computes mean squared error on the GPU.
*/
func (trainingOps *MetalTrainingOps) MSELoss(predictions, targets []float64) ([]float64, error) {
	if len(predictions) == 0 {
		return []float64{0}, nil
	}

	output := make([]float32, 1)
	prediction32 := toFloat32(predictions)
	target32 := toFloat32(targets)
	rc := C.metal_train_mse_loss(
		(*C.float)(unsafe.Pointer(&prediction32[0])),
		(*C.float)(unsafe.Pointer(&target32[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_train_mse_loss failed (rc=%d)", rc)
	}

	return toFloat64(output), nil
}

/*
CrossEntropyLoss computes softmax cross-entropy on the GPU.
*/
func (trainingOps *MetalTrainingOps) CrossEntropyLoss(
	logits, targets []float64,
) ([]float64, error) {
	output := make([]float32, 1)
	logits32 := toFloat32(logits)
	target32 := toFloat32(targets)
	rc := C.metal_train_cross_entropy_loss(
		(*C.float)(unsafe.Pointer(&logits32[0])),
		(*C.float)(unsafe.Pointer(&target32[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(len(logits)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_train_cross_entropy_loss failed (rc=%d)", rc)
	}

	return toFloat64(output), nil
}

/*
MSEGrad computes the MSE gradient on the GPU.
*/
func (trainingOps *MetalTrainingOps) MSEGrad(predictions, targets []float64) ([]float64, error) {
	if len(predictions) == 0 {
		return []float64{}, nil
	}

	output := make([]float32, len(predictions))
	prediction32 := toFloat32(predictions)
	target32 := toFloat32(targets)
	rc := C.metal_train_mse_grad(
		(*C.float)(unsafe.Pointer(&prediction32[0])),
		(*C.float)(unsafe.Pointer(&target32[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_train_mse_grad failed (rc=%d)", rc)
	}

	return toFloat64(output), nil
}

/*
CrossEntropyGrad computes softmax cross-entropy gradient on the GPU.
*/
func (trainingOps *MetalTrainingOps) CrossEntropyGrad(
	logits, targets []float64,
) ([]float64, error) {
	if len(logits) == 0 {
		return []float64{}, nil
	}

	output := make([]float32, len(logits))
	logits32 := toFloat32(logits)
	target32 := toFloat32(targets)
	rc := C.metal_train_cross_entropy_grad(
		(*C.float)(unsafe.Pointer(&logits32[0])),
		(*C.float)(unsafe.Pointer(&target32[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(len(logits)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_train_cross_entropy_grad failed (rc=%d)", rc)
	}

	return toFloat64(output), nil
}

/*
Accuracy computes single-sample argmax equality on the GPU.
*/
func (trainingOps *MetalTrainingOps) Accuracy(predictions, targets []float64) ([]float64, error) {
	output := make([]float32, 1)
	prediction32 := toFloat32(predictions)
	target32 := toFloat32(targets)
	rc := C.metal_bench_accuracy(
		(*C.float)(unsafe.Pointer(&prediction32[0])),
		(*C.float)(unsafe.Pointer(&target32[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_bench_accuracy failed (rc=%d)", rc)
	}

	return toFloat64(output), nil
}

/*
F1Counts computes TP, FP, and FN counts on the GPU.
*/
func (trainingOps *MetalTrainingOps) F1Counts(predictions, targets []float64) ([]float64, error) {
	output := make([]float32, 3)

	if len(predictions) == 0 {
		return toFloat64(output), nil
	}

	prediction32 := toFloat32(predictions)
	target32 := toFloat32(targets)
	rc := C.metal_bench_f1_counts(
		(*C.float)(unsafe.Pointer(&prediction32[0])),
		(*C.float)(unsafe.Pointer(&target32[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(len(predictions)),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_bench_f1_counts failed (rc=%d)", rc)
	}

	return toFloat64(output), nil
}
