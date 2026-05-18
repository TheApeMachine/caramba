//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (tensorBackend *TensorBackend) applyMQA(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Tensor,
) (computetensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	queryShape, keyValueShape, err := attentionVariantShapes(node, inputs, "MQA")
	if err != nil {
		return nil, err
	}

	if keyValueShape[1] != 1 || queryShape[0] != keyValueShape[0] ||
		queryShape[2] != keyValueShape[2] || queryShape[3] != keyValueShape[3] {
		return nil, fmt.Errorf("metal tensor: MQA node %q query/key/value shape mismatch", node.ID)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	attentionOps, err := tensorBackend.attention()
	if err != nil {
		return nil, err
	}

	return attentionOps.MQATensor(
		inputs[0],
		inputs[1],
		inputs[2],
		outputShape,
		queryShape[0],
		queryShape[1],
		queryShape[2],
		queryShape[3],
	)
}

func (tensorBackend *TensorBackend) applySlidingWindowAttention(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Tensor,
) (computetensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	queryShape, keyValueShape, err := attentionVariantShapes(node, inputs, "sliding_window")
	if err != nil {
		return nil, err
	}

	if queryShape[0] != keyValueShape[0] || queryShape[1] != keyValueShape[1] ||
		queryShape[2] != keyValueShape[2] || queryShape[3] != keyValueShape[3] ||
		!inputs[1].Shape().Equal(inputs[2].Shape()) {
		return nil, fmt.Errorf(
			"metal tensor: sliding_window node %q query/key/value shape mismatch",
			node.ID,
		)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	attentionOps, err := tensorBackend.attention()
	if err != nil {
		return nil, err
	}

	return attentionOps.SlidingWindowTensor(
		inputs[0],
		inputs[1],
		inputs[2],
		outputShape,
		queryShape[0],
		queryShape[1],
		queryShape[2],
		queryShape[3],
		intConfig(node, "window", 0),
	)
}

func attentionVariantShapes(
	node executor.NodeSpec,
	inputs []computetensor.Tensor,
	operation string,
) ([]int, []int, error) {
	if len(inputs) != 3 {
		return nil, nil, fmt.Errorf("metal tensor: %s node %q requires 3 inputs", operation, node.ID)
	}

	queryShape := inputs[0].Shape().Dims()
	keyValueShape := inputs[1].Shape().Dims()

	if len(queryShape) != 4 {
		return nil, nil, fmt.Errorf("metal tensor: %s node %q query must be rank 4", operation, node.ID)
	}

	if len(keyValueShape) != 4 || !inputs[1].Shape().Equal(inputs[2].Shape()) {
		return nil, nil, fmt.Errorf(
			"metal tensor: %s node %q key/value must be matching rank 4 tensors",
			operation,
			node.ID,
		)
	}

	return queryShape, keyValueShape, nil
}
