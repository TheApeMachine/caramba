//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "optimizer.h"
import "C"

import (
	"context"
	"fmt"
	"math"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (tensorBackend *TensorBackend) applyOptimizerStep(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 2 {
		return nil, fmt.Errorf("metal tensor: optimizer node %q requires 2 inputs", node.ID)
	}

	params, grads, err := trainingPair(inputs[0], inputs[1])
	if err != nil {
		return nil, err
	}

	if params.Len() == 0 {
		return nil, fmt.Errorf("metal tensor: optimizer node %q requires non-empty inputs", node.ID)
	}

	if err := tensorBackend.initOptimizerKernels(); err != nil {
		return nil, err
	}

	output, err := tensorBackend.runtime.NewFloat32Tensor(params.shape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := tensorBackend.dispatchOptimizerStep(node, params, grads, output)
	if rc != 0 {
		_ = output.Close()

		return nil, metalOptimizerError(rc, string(node.Op))
	}

	return output, nil
}

func (tensorBackend *TensorBackend) dispatchOptimizerStep(
	node executor.NodeSpec,
	params *Tensor,
	grads *Tensor,
	output *Tensor,
) C.int {
	operation := executor.NormalizeOperation(node.Op)
	elementCount := C.int(params.Len())

	switch operation {
	case "train.optimizer.adam", "optimizer.adam":
		beta1 := floatConfig(node, "beta1", 0.9)
		beta2 := floatConfig(node, "beta2", 0.999)
		return C.metal_optimizer_adam_tensor(
			params.buffer, grads.buffer, output.buffer, elementCount,
			C.double(beta1), C.double(beta2),
			C.double(adamStepLearningRate(node, beta1, beta2)),
			C.double(floatConfig(node, "eps", 1e-8)),
		)
	case "train.optimizer.adamw", "optimizer.adamw":
		beta1 := floatConfig(node, "beta1", 0.9)
		beta2 := floatConfig(node, "beta2", 0.999)
		learningRate := floatConfig(node, "lr", 1e-3)
		return C.metal_optimizer_adamw_tensor(
			params.buffer, grads.buffer, output.buffer, elementCount,
			C.double(beta1), C.double(beta2),
			C.double(adamStepLearningRate(node, beta1, beta2)),
			C.double(floatConfig(node, "eps", 1e-8)),
			C.double(learningRate*floatConfig(node, "wd", 0)),
		)
	case "train.optimizer.adamax", "optimizer.adamax":
		beta1 := floatConfig(node, "beta1", 0.9)
		learningRate := floatConfig(node, "lr", 2e-3) / (1 - beta1)
		return C.metal_optimizer_adamax_tensor(
			params.buffer, grads.buffer, output.buffer, elementCount,
			C.double(beta1), C.double(floatConfig(node, "beta2", 0.999)),
			C.double(learningRate), C.double(floatConfig(node, "eps", 1e-8)),
		)
	case "train.optimizer.sgd", "optimizer.sgd":
		return C.metal_optimizer_sgd_tensor(
			params.buffer, grads.buffer, output.buffer, elementCount,
			C.double(floatConfig(node, "lr", 1e-3)),
			C.double(floatConfig(node, "wd", 0)),
			C.double(floatConfig(node, "momentum", 0)),
			boolInt(boolConfig(node, "nesterov", false)),
		)
	case "train.optimizer.lion", "optimizer.lion":
		return C.metal_optimizer_lion_tensor(
			params.buffer, grads.buffer, output.buffer, elementCount,
			C.double(floatConfig(node, "lr", 1e-4)),
			C.double(floatConfig(node, "beta1", 0.9)),
			C.double(floatConfig(node, "beta2", 0.99)),
			C.double(floatConfig(node, "wd", 0)),
		)
	case "train.optimizer.rmsprop", "optimizer.rmsprop":
		return C.metal_optimizer_rmsprop_tensor(
			params.buffer, grads.buffer, output.buffer, elementCount,
			C.double(floatConfig(node, "lr", 1e-2)),
			C.double(floatConfig(node, "alpha", 0.99)),
			C.double(floatConfig(node, "eps", 1e-8)),
			C.double(floatConfig(node, "momentum", 0)),
			C.double(floatConfig(node, "wd", 0)),
			boolInt(boolConfig(node, "centered", false)),
		)
	case "train.optimizer.hebbian", "optimizer.hebbian":
		return C.metal_optimizer_hebbian_tensor(
			params.buffer, grads.buffer, output.buffer, elementCount,
			C.double(floatConfig(node, "lr", 1e-3)),
			C.double(floatConfig(node, "max_norm", 0)),
		)
	case "train.optimizer.lars", "optimizer.lars":
		return C.metal_optimizer_lars_tensor(
			params.buffer, grads.buffer, output.buffer, elementCount,
			C.double(floatConfig(node, "lr", 1e-2)),
			C.double(floatConfig(node, "eta", 1e-3)),
			C.double(floatConfig(node, "momentum", 0.9)),
			C.double(floatConfig(node, "wd", 0)),
			C.double(floatConfig(node, "eps", 1e-8)),
		)
	case "train.optimizer.lamb", "optimizer.lamb":
		beta1 := floatConfig(node, "beta1", 0.9)
		beta2 := floatConfig(node, "beta2", 0.999)
		return C.metal_optimizer_lamb_tensor(
			params.buffer, grads.buffer, output.buffer, elementCount,
			C.double(floatConfig(node, "lr", 1e-3)),
			C.double(beta1), C.double(beta2),
			C.double(floatConfig(node, "eps", 1e-6)),
			C.double(floatConfig(node, "wd", 0)),
			C.double(1/(1-beta1)), C.double(1/(1-beta2)),
		)
	case "train.optimizer.adagrad", "optimizer.adagrad":
		return C.metal_optimizer_adagrad_tensor(
			params.buffer, grads.buffer, output.buffer, elementCount,
			C.double(floatConfig(node, "lr", 1e-2)),
			C.double(floatConfig(node, "eps", 1e-10)),
			C.double(floatConfig(node, "wd", 0)),
		)
	case "train.optimizer.adadelta", "optimizer.adadelta":
		return C.metal_optimizer_adadelta_tensor(
			params.buffer, grads.buffer, output.buffer, elementCount,
			C.double(floatConfig(node, "rho", 0.9)),
			C.double(floatConfig(node, "eps", 1e-6)),
			C.double(floatConfig(node, "wd", 0)),
		)
	case "train.optimizer.lbfgs", "optimizer.lbfgs":
		return C.metal_optimizer_lbfgs_tensor(
			params.buffer, grads.buffer, output.buffer, elementCount,
			C.double(floatConfig(node, "lr", 1.0)),
			boolInt(boolConfig(node, "line_search", false)),
			C.double(floatConfig(node, "c1", 1e-4)),
		)
	}

	return -1
}

func (tensorBackend *TensorBackend) initOptimizerKernels() error {
	path := C.CString(metalLibrary(nil, "optimizer.metallib"))
	defer C.free(unsafe.Pointer(path))

	rc := C.metal_optimizer_init(path)
	if rc != 0 {
		return metalOptimizerError(rc, "init")
	}

	return nil
}

func adamStepLearningRate(node executor.NodeSpec, beta1 float64, beta2 float64) float64 {
	learningRate := floatConfig(node, "lr", 1e-3)

	return learningRate * math.Sqrt(1-beta2) / (1 - beta1)
}
