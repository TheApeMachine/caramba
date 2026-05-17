//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "optimizer.h"
import "C"

import (
	"context"
	"fmt"
	"math"
	"slices"
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

	optimizerState, err := tensorBackend.optimizerState(node, params)

	if err != nil {
		return nil, err
	}

	output, err := tensorBackend.runtime.NewFloat32Tensor(params.shape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := tensorBackend.dispatchOptimizerStep(node, optimizerState, params, grads, output)
	if rc != 0 {
		_ = output.Close()

		return nil, metalOptimizerError(rc, string(node.Op))
	}

	if err := optimizerState.commit(node, params, grads); err != nil {
		_ = output.Close()

		return nil, err
	}

	return output, nil
}

func (tensorBackend *TensorBackend) dispatchOptimizerStep(
	node executor.NodeSpec,
	optimizerState *residentOptimizerState,
	params *Tensor,
	grads *Tensor,
	output *Tensor,
) C.int {
	operation := string(executor.NormalizeOperation(node.Op))
	elementCount := C.int(params.Len())
	nextStep := optimizerState.step + 1

	switch operation {
	case "train.optimizer.adam", "optimizer.adam":
		beta1 := floatConfig(node, "beta1", 0.9)
		beta2 := floatConfig(node, "beta2", 0.999)
		return C.metal_optimizer_adam_tensor(
			params.buffer, grads.buffer,
			optimizerState.buffer("moment"),
			optimizerState.buffer("variance"),
			output.buffer, elementCount,
			C.double(beta1), C.double(beta2),
			C.double(adamStepLearningRate(node, beta1, beta2, nextStep)),
			C.double(floatConfig(node, "eps", 1e-8)),
		)
	case "train.optimizer.adamw", "optimizer.adamw":
		beta1 := floatConfig(node, "beta1", 0.9)
		beta2 := floatConfig(node, "beta2", 0.999)
		learningRate := floatConfig(node, "lr", 1e-3)
		return C.metal_optimizer_adamw_tensor(
			params.buffer, grads.buffer,
			optimizerState.buffer("moment"),
			optimizerState.buffer("variance"),
			output.buffer, elementCount,
			C.double(beta1), C.double(beta2),
			C.double(adamStepLearningRate(node, beta1, beta2, nextStep)),
			C.double(floatConfig(node, "eps", 1e-8)),
			C.double(learningRate*floatConfig(node, "wd", 0)),
		)
	case "train.optimizer.adamax", "optimizer.adamax":
		beta1 := floatConfig(node, "beta1", 0.9)
		learningRate := floatConfig(node, "lr", 2e-3) /
			(1 - math.Pow(beta1, float64(nextStep)))
		return C.metal_optimizer_adamax_tensor(
			params.buffer, grads.buffer,
			optimizerState.buffer("moment"),
			optimizerState.buffer("infinity_norm"),
			output.buffer, elementCount,
			C.double(beta1), C.double(floatConfig(node, "beta2", 0.999)),
			C.double(learningRate), C.double(floatConfig(node, "eps", 1e-8)),
		)
	case "train.optimizer.sgd", "optimizer.sgd":
		return C.metal_optimizer_sgd_tensor(
			params.buffer, grads.buffer,
			optimizerState.buffer("velocity"),
			output.buffer, elementCount,
			C.double(floatConfig(node, "lr", 1e-3)),
			C.double(floatConfig(node, "wd", 0)),
			C.double(floatConfig(node, "momentum", 0)),
			boolInt(boolConfig(node, "nesterov", false)),
		)
	case "train.optimizer.lion", "optimizer.lion":
		return C.metal_optimizer_lion_tensor(
			params.buffer, grads.buffer,
			optimizerState.buffer("moment"),
			output.buffer, elementCount,
			C.double(floatConfig(node, "lr", 1e-4)),
			C.double(floatConfig(node, "beta1", 0.9)),
			C.double(floatConfig(node, "beta2", 0.99)),
			C.double(floatConfig(node, "wd", 0)),
		)
	case "train.optimizer.rmsprop", "optimizer.rmsprop":
		return C.metal_optimizer_rmsprop_tensor(
			params.buffer, grads.buffer,
			optimizerState.buffer("square_average"),
			optimizerState.buffer("momentum_buffer"),
			optimizerState.buffer("grad_average"),
			output.buffer, elementCount,
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
			params.buffer, grads.buffer,
			optimizerState.buffer("velocity"),
			output.buffer, elementCount,
			C.double(floatConfig(node, "lr", 1e-2)),
			C.double(floatConfig(node, "eta", 1e-3)),
			C.double(floatConfig(node, "momentum", 0.9)),
			C.double(floatConfig(node, "wd", 0)),
			C.double(floatConfig(node, "eps", 1e-8)),
		)
	case "train.optimizer.lamb", "optimizer.lamb":
		beta1 := floatConfig(node, "beta1", 0.9)
		beta2 := floatConfig(node, "beta2", 0.999)
		biasCorrection1Inv := 1 / (1 - math.Pow(beta1, float64(nextStep)))
		biasCorrection2Inv := 1 / (1 - math.Pow(beta2, float64(nextStep)))

		return C.metal_optimizer_lamb_tensor(
			params.buffer, grads.buffer,
			optimizerState.buffer("moment"),
			optimizerState.buffer("variance"),
			output.buffer, elementCount,
			C.double(floatConfig(node, "lr", 1e-3)),
			C.double(beta1), C.double(beta2),
			C.double(floatConfig(node, "eps", 1e-6)),
			C.double(floatConfig(node, "wd", 0)),
			C.double(biasCorrection1Inv), C.double(biasCorrection2Inv),
		)
	case "train.optimizer.adagrad", "optimizer.adagrad":
		return C.metal_optimizer_adagrad_tensor(
			params.buffer, grads.buffer,
			optimizerState.buffer("accumulator"),
			output.buffer, elementCount,
			C.double(floatConfig(node, "lr", 1e-2)),
			C.double(floatConfig(node, "eps", 1e-10)),
			C.double(floatConfig(node, "wd", 0)),
		)
	case "train.optimizer.adadelta", "optimizer.adadelta":
		return C.metal_optimizer_adadelta_tensor(
			params.buffer, grads.buffer,
			optimizerState.buffer("grad_average"),
			optimizerState.buffer("delta_average"),
			output.buffer, elementCount,
			C.double(floatConfig(node, "rho", 0.9)),
			C.double(floatConfig(node, "eps", 1e-6)),
			C.double(floatConfig(node, "wd", 0)),
		)
	case "train.optimizer.lbfgs", "optimizer.lbfgs":
		return C.metal_optimizer_lbfgs_tensor(
			params.buffer, grads.buffer,
			optimizerState.buffer("state_history"),
			optimizerState.buffer("grad_history"),
			optimizerState.buffer("rho_history"),
			optimizerState.buffer("head"),
			optimizerState.buffer("history_count"),
			optimizerState.buffer("previous_params"),
			optimizerState.buffer("previous_grads"),
			output.buffer,
			boolInt(optimizerState.hasPrevious),
			elementCount,
			C.int(optimizerState.historySize),
			C.double(floatConfig(node, "lr", 1.0)),
			boolInt(boolConfig(node, "line_search", false)),
			C.double(floatConfig(node, "c1", 1e-4)),
		)
	}

	return -1
}

type residentOptimizerState struct {
	operation   string
	shape       []int
	historySize int
	step        int
	hasPrevious bool
	tensors     map[string]*Tensor
}

func (tensorBackend *TensorBackend) optimizerState(
	node executor.NodeSpec,
	params *Tensor,
) (*residentOptimizerState, error) {
	operation := string(executor.NormalizeOperation(node.Op))
	historySize := optimizerHistorySize(node)
	shape := params.Shape().Dims()

	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.optimizerStates == nil {
		tensorBackend.optimizerStates = make(map[string]*residentOptimizerState)
	}

	stateKey := node.ID + ":" + operation
	if state := tensorBackend.optimizerStates[stateKey]; state != nil &&
		state.matches(operation, shape, historySize) {
		return state, nil
	}

	if state := tensorBackend.optimizerStates[stateKey]; state != nil {
		if err := state.close(); err != nil {
			return nil, err
		}
	}

	state, err := tensorBackend.newOptimizerState(operation, shape, historySize)

	if err != nil {
		return nil, err
	}

	tensorBackend.optimizerStates[stateKey] = state

	return state, nil
}

func (tensorBackend *TensorBackend) newOptimizerState(
	operation string,
	shape []int,
	historySize int,
) (*residentOptimizerState, error) {
	state := &residentOptimizerState{
		operation:   operation,
		shape:       slices.Clone(shape),
		historySize: historySize,
		tensors:     make(map[string]*Tensor),
	}

	for name, elementCount := range optimizerStateSpec(operation, shape, historySize) {
		storageMode := MetalStorageModePrivate

		if name == "head" || name == "history_count" {
			storageMode = MetalStorageModeShared
		}

		value, err := tensorBackend.newZeroOptimizerStateTensor(elementCount, storageMode)

		if err != nil {
			_ = state.close()

			return nil, err
		}

		state.tensors[name] = value
	}

	return state, nil
}

func (tensorBackend *TensorBackend) newZeroOptimizerStateTensor(
	elementCount int,
	storageMode MetalStorageMode,
) (*Tensor, error) {
	shape, err := computetensor.NewShape([]int{elementCount})

	if err != nil {
		return nil, err
	}

	value, err := tensorBackend.runtime.newTensor(
		shape,
		computetensor.Float32,
		storageMode,
		MetalAllocationPersistent,
	)

	if err != nil {
		return nil, err
	}

	rc := C.metal_optimizer_zero_tensor(value.buffer, C.int(elementCount))

	if rc != 0 {
		_ = value.Close()

		return nil, metalOptimizerError(rc, "optimizer state zero")
	}

	return value, nil
}

func optimizerStateSpec(operation string, shape []int, historySize int) map[string]int {
	elementCount := shapeElementCount(shape)

	switch operation {
	case "train.optimizer.adam", "optimizer.adam",
		"train.optimizer.adamw", "optimizer.adamw",
		"train.optimizer.lamb", "optimizer.lamb":
		return map[string]int{"moment": elementCount, "variance": elementCount}
	case "train.optimizer.adamax", "optimizer.adamax":
		return map[string]int{"moment": elementCount, "infinity_norm": elementCount}
	case "train.optimizer.sgd", "optimizer.sgd",
		"train.optimizer.lars", "optimizer.lars":
		return map[string]int{"velocity": elementCount}
	case "train.optimizer.lion", "optimizer.lion":
		return map[string]int{"moment": elementCount}
	case "train.optimizer.rmsprop", "optimizer.rmsprop":
		return map[string]int{
			"square_average":  elementCount,
			"momentum_buffer": elementCount,
			"grad_average":    elementCount,
		}
	case "train.optimizer.adagrad", "optimizer.adagrad":
		return map[string]int{"accumulator": elementCount}
	case "train.optimizer.adadelta", "optimizer.adadelta":
		return map[string]int{
			"grad_average":  elementCount,
			"delta_average": elementCount,
		}
	case "train.optimizer.lbfgs", "optimizer.lbfgs":
		return map[string]int{
			"state_history":   elementCount * historySize,
			"grad_history":    elementCount * historySize,
			"rho_history":     historySize,
			"head":            1,
			"history_count":   1,
			"previous_params": elementCount,
			"previous_grads":  elementCount,
		}
	default:
		return map[string]int{}
	}
}

func optimizerHistorySize(node executor.NodeSpec) int {
	historySize := intConfigAny(node, 10, "history_size", "hist_size")

	if historySize <= 0 {
		return 10
	}

	return historySize
}

func shapeElementCount(shape []int) int {
	elementCount := 1

	for _, dimension := range shape {
		elementCount *= dimension
	}

	return elementCount
}

func (optimizerState *residentOptimizerState) matches(
	operation string,
	shape []int,
	historySize int,
) bool {
	return optimizerState.operation == operation &&
		optimizerState.historySize == historySize &&
		slices.Equal(optimizerState.shape, shape)
}

func (optimizerState *residentOptimizerState) buffer(name string) unsafe.Pointer {
	if optimizerState == nil || optimizerState.tensors[name] == nil {
		return nil
	}

	return optimizerState.tensors[name].buffer
}

func (optimizerState *residentOptimizerState) commit(
	node executor.NodeSpec,
	params *Tensor,
	grads *Tensor,
) error {
	operation := string(executor.NormalizeOperation(node.Op))

	if operation != "train.optimizer.lbfgs" && operation != "optimizer.lbfgs" {
		optimizerState.step++

		return nil
	}

	if err := optimizerState.copyTensor("previous_params", params); err != nil {
		return err
	}

	if err := optimizerState.copyTensor("previous_grads", grads); err != nil {
		return err
	}

	optimizerState.hasPrevious = true
	optimizerState.step++

	return nil
}

func (optimizerState *residentOptimizerState) copyTensor(name string, source *Tensor) error {
	destination := optimizerState.tensors[name]

	if destination == nil {
		return fmt.Errorf("metal tensor: optimizer state %q is not allocated", name)
	}

	rc := C.metal_optimizer_copy_tensor(source.buffer, destination.buffer, C.int(source.Len()))

	if rc == 0 {
		return nil
	}

	return metalOptimizerError(rc, "optimizer state copy")
}

func (optimizerState *residentOptimizerState) close() error {
	if optimizerState == nil {
		return nil
	}

	var closeErr error

	for key, value := range optimizerState.tensors {
		if value == nil {
			continue
		}

		if err := value.Close(); err != nil && closeErr == nil {
			closeErr = err
		}

		delete(optimizerState.tensors, key)
	}

	return closeErr
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

func adamStepLearningRate(
	node executor.NodeSpec,
	beta1 float64,
	beta2 float64,
	step int,
) float64 {
	learningRate := floatConfig(node, "lr", 1e-3)

	return learningRate * math.Sqrt(1-math.Pow(beta2, float64(step))) /
		(1 - math.Pow(beta1, float64(step)))
}
