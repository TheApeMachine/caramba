//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (tensorBackend *TensorBackend) applyActiveFreeEnergy(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 2 {
		return nil, fmt.Errorf("metal tensor: active_inference.free_energy node %q requires 2 inputs", node.ID)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	activeOps, err := tensorBackend.activeInference()
	if err != nil {
		return nil, err
	}

	return activeOps.FreeEnergyTensor(inputs[0], inputs[1], outputShape)
}

func (tensorBackend *TensorBackend) applyActiveBeliefUpdate(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("metal tensor: active_inference.belief_update node %q requires 3 inputs", node.ID)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	activeOps, err := tensorBackend.activeInference()
	if err != nil {
		return nil, err
	}

	return activeOps.BeliefUpdateTensor(
		inputs[0],
		inputs[1],
		inputs[2],
		outputShape,
		activeLearningRate(node),
	)
}

func (tensorBackend *TensorBackend) applyActivePrecisionWeight(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 2 {
		return nil, fmt.Errorf("metal tensor: active_inference.precision_weight node %q requires 2 inputs", node.ID)
	}

	activeOps, err := tensorBackend.activeInference()
	if err != nil {
		return nil, err
	}

	return activeOps.PrecisionWeightTensor(inputs[0], inputs[1])
}

func (tensorBackend *TensorBackend) applyActiveExpectedFreeEnergy(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: active_inference.expected_free_energy node %q requires 1 input", node.ID)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	policyCount := outputShape.Len()
	outcomeCount := intConfig(node, "outcome_count", 0)
	if outcomeCount == 0 && policyCount > 0 {
		outcomeCount = inputs[0].Shape().Len() / policyCount
	}

	activeOps, err := tensorBackend.activeInference()
	if err != nil {
		return nil, err
	}

	return activeOps.ExpectedFreeEnergyTensor(inputs[0], outputShape, outcomeCount, policyCount)
}

func (tensorBackend *TensorBackend) activeInference() (*ActiveInferenceOps, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.activeOps != nil {
		return tensorBackend.activeOps, nil
	}

	activeOps, err := NewActiveInferenceOps(metalLibrary(nil, "active_inference.metallib"))
	if err != nil {
		return nil, err
	}

	activeOps.runtime = tensorBackend.runtime
	tensorBackend.activeOps = activeOps

	return activeOps, nil
}

func activeLearningRate(node executor.NodeSpec) float32 {
	learningRate := floatConfig(node, "learning_rate", 0)
	if learningRate != 0 {
		return float32(learningRate)
	}

	learningRate = floatConfig(node, "lr", 0)
	if learningRate != 0 {
		return float32(learningRate)
	}

	rawSteps := intConfig(node, "raw_lr_steps", 0)
	if rawSteps == 0 {
		rawSteps = intConfig(node, "lr_steps", 1)
	}

	return float32(rawSteps) * 1e-4
}
