//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "metal_kernel_math.h"
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

/*
MetalTrainingOps dispatches training losses, gradients, and benchmark kernels through
math.metallib. The kernels live beside MathOps because they share the same primitive
reductions and elementwise math pipeline.
*/
type MetalTrainingOps struct {
	mu       sync.Mutex
	metallib string
	closed   bool
}

var metalTrainingOpsShutdownMu sync.Mutex

/*
NewTrainingOps creates and initializes MetalTrainingOps.
*/
func NewTrainingOps(metallib string) (*MetalTrainingOps, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_math_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_math_init failed (rc=%d): check %q exists", rc, metallib)
	}

	trainingOps := &MetalTrainingOps{metallib: metallib}
	runtime.SetFinalizer(trainingOps, func(trainingOps *MetalTrainingOps) {
		_ = trainingOps.Close()
	})

	return trainingOps, nil
}

/*
Close releases Metal math resources initialized for MetalTrainingOps.
*/
func (trainingOps *MetalTrainingOps) Close() error {
	trainingOps.mu.Lock()
	defer trainingOps.mu.Unlock()

	if trainingOps.closed {
		return nil
	}

	trainingOps.closed = true
	trainingOps.metallib = ""
	runtime.SetFinalizer(trainingOps, nil)

	metalTrainingOpsShutdownMu.Lock()
	defer metalTrainingOpsShutdownMu.Unlock()

	if rc := C.metal_math_shutdown(); rc != 0 {
		return fmt.Errorf("metal_math_shutdown failed (rc=%d)", rc)
	}

	return nil
}

/*
MSELoss computes mean squared error on the GPU.
*/
func (trainingOps *MetalTrainingOps) MSELoss(predictions, targets []float64) ([]float64, error) {
	if len(predictions) != len(targets) {
		return nil, fmt.Errorf(
			"MetalTrainingOps.MSELoss: length mismatch predictions=%d targets=%d",
			len(predictions), len(targets),
		)
	}

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
	if len(logits) == 0 || len(targets) == 0 {
		return nil, fmt.Errorf("MetalTrainingOps.CrossEntropyLoss: logits and targets must be non-empty")
	}

	if len(logits) != len(targets) {
		return nil, fmt.Errorf(
			"MetalTrainingOps.CrossEntropyLoss: length mismatch logits=%d targets=%d",
			len(logits), len(targets),
		)
	}

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
	if len(predictions) != len(targets) {
		return nil, fmt.Errorf(
			"MetalTrainingOps.MSEGrad: length mismatch predictions=%d targets=%d",
			len(predictions), len(targets),
		)
	}

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
	if len(logits) != len(targets) {
		return nil, fmt.Errorf(
			"MetalTrainingOps.CrossEntropyGrad: length mismatch logits=%d targets=%d",
			len(logits), len(targets),
		)
	}

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
	if len(predictions) == 0 || len(targets) == 0 {
		return nil, fmt.Errorf("MetalTrainingOps.Accuracy: predictions and targets must be non-empty")
	}

	if len(predictions) != len(targets) {
		return nil, fmt.Errorf(
			"MetalTrainingOps.Accuracy: length mismatch predictions=%d targets=%d",
			len(predictions), len(targets),
		)
	}

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

	if len(predictions) != len(targets) {
		return nil, fmt.Errorf(
			"MetalTrainingOps.F1Counts: length mismatch predictions=%d targets=%d",
			len(predictions), len(targets),
		)
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
