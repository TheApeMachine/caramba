//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (tensorBackend *TensorBackend) applyTrainingPair(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
	operation func(computetensor.Float64Tensor, computetensor.Float64Tensor) (computetensor.Float64Tensor, error),
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 2 {
		return nil, fmt.Errorf("metal tensor: training node %q requires 2 inputs", node.ID)
	}

	return operation(inputs[0], inputs[1])
}

func (tensorBackend *TensorBackend) applyMSELoss(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return tensorBackend.applyTrainingPair(ctx, node, inputs, mathOps.MSELossTensor)
}

func (tensorBackend *TensorBackend) applyCrossEntropyLoss(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return tensorBackend.applyTrainingPair(ctx, node, inputs, mathOps.CrossEntropyLossTensor)
}

func (tensorBackend *TensorBackend) applyMSEGrad(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return tensorBackend.applyTrainingPair(ctx, node, inputs, mathOps.MSEGradTensor)
}

func (tensorBackend *TensorBackend) applyCrossEntropyGrad(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return tensorBackend.applyTrainingPair(ctx, node, inputs, mathOps.CrossEntropyGradTensor)
}

func (tensorBackend *TensorBackend) applyAccuracy(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return tensorBackend.applyTrainingPair(ctx, node, inputs, mathOps.AccuracyTensor)
}

func (tensorBackend *TensorBackend) applyPerplexity(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return tensorBackend.applyTrainingPair(ctx, node, inputs, mathOps.PerplexityTensor)
}

func (tensorBackend *TensorBackend) applyF1(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return tensorBackend.applyTrainingPair(ctx, node, inputs, mathOps.F1Tensor)
}
