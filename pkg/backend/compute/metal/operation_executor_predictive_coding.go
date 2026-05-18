//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (tensorBackend *TensorBackend) applyPredictivePrediction(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Tensor,
) (computetensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 2 {
		return nil, fmt.Errorf("metal tensor: predictive_coding.prediction node %q requires 2 inputs", node.ID)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	predictiveOps, err := tensorBackend.predictiveCoding()
	if err != nil {
		return nil, err
	}

	return predictiveOps.PredictionTensor(inputs[0], inputs[1], outputShape)
}

func (tensorBackend *TensorBackend) applyPredictivePredictionError(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Tensor,
) (computetensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) < 2 || len(inputs) > 3 {
		return nil, fmt.Errorf("metal tensor: predictive_coding.prediction_error node %q requires 2 or 3 inputs", node.ID)
	}

	var precision computetensor.Tensor
	if len(inputs) == 3 {
		precision = inputs[2]
	}

	predictiveOps, err := tensorBackend.predictiveCoding()
	if err != nil {
		return nil, err
	}

	return predictiveOps.PredictionErrorTensor(inputs[0], inputs[1], precision)
}

func (tensorBackend *TensorBackend) applyPredictiveUpdateRepresentation(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Tensor,
) (computetensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 5 {
		return nil, fmt.Errorf("metal tensor: predictive_coding.update_representation node %q requires 5 inputs", node.ID)
	}

	predictiveOps, err := tensorBackend.predictiveCoding()
	if err != nil {
		return nil, err
	}

	return predictiveOps.UpdateRepresentationTensor(
		inputs[0],
		inputs[1],
		inputs[2],
		inputs[3],
		inputs[4],
	)
}

func (tensorBackend *TensorBackend) applyPredictiveUpdateWeights(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Tensor,
) (computetensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 4 {
		return nil, fmt.Errorf("metal tensor: predictive_coding.update_weights node %q requires 4 inputs", node.ID)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	predictiveOps, err := tensorBackend.predictiveCoding()
	if err != nil {
		return nil, err
	}

	return predictiveOps.UpdateWeightsTensor(inputs[0], inputs[1], inputs[2], outputShape, inputs[3])
}

func (tensorBackend *TensorBackend) predictiveCoding() (*MetalPredictiveCodingOps, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.predictiveOps != nil {
		return tensorBackend.predictiveOps, nil
	}

	predictiveOps, err := NewPredictiveCodingOps(metalLibrary(nil, "predictive_coding.metallib"))
	if err != nil {
		return nil, err
	}

	predictiveOps.runtime, err = tensorBackend.sharedRuntime(predictiveOps.runtime)

	if err != nil {
		return nil, err
	}

	tensorBackend.predictiveOps = predictiveOps

	return predictiveOps, nil
}
