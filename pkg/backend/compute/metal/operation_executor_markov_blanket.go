//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (tensorBackend *TensorBackend) applyMarkovPartition(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 2 && len(inputs) != 5 {
		return nil, fmt.Errorf("metal tensor: markov_blanket.partition node %q requires 2 or 5 inputs", node.ID)
	}

	partition, err := markovPartitionShape(node, inputs[0].Shape().Len())
	if err != nil {
		return nil, err
	}

	marks, closeMarks, err := tensorBackend.markovPartitionMasks(node, inputs)
	if err != nil {
		return nil, err
	}
	if closeMarks {
		defer func() {
			_ = marks.Close()
		}()
	}

	markovOps, err := tensorBackend.markovBlanket()
	if err != nil {
		return nil, err
	}

	return markovOps.PartitionTensor(
		inputs[0],
		marks,
		partition.outputShape,
		partition.sensoryCount,
		partition.activeCount,
		partition.internalCount,
		partition.externalCount,
	)
}

func (tensorBackend *TensorBackend) applyMarkovFlowInternal(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("metal tensor: markov_blanket.flow_internal node %q requires 3 inputs", node.ID)
	}

	outputShape, err := markovFlowShape(node, "markov_blanket.flow_internal")
	if err != nil {
		return nil, err
	}

	markovOps, err := tensorBackend.markovBlanket()
	if err != nil {
		return nil, err
	}

	return markovOps.FlowInternalTensor(inputs[0], inputs[1], inputs[2], outputShape)
}

func (tensorBackend *TensorBackend) applyMarkovFlowActive(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("metal tensor: markov_blanket.flow_active node %q requires 3 inputs", node.ID)
	}

	outputShape, err := markovFlowShape(node, "markov_blanket.flow_active")
	if err != nil {
		return nil, err
	}

	markovOps, err := tensorBackend.markovBlanket()
	if err != nil {
		return nil, err
	}

	return markovOps.FlowActiveTensor(inputs[0], inputs[1], inputs[2], outputShape)
}

func (tensorBackend *TensorBackend) applyMarkovMutualInformation(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 2 {
		return nil, fmt.Errorf("metal tensor: markov_blanket.mutual_information node %q requires 2 inputs", node.ID)
	}

	xDimensions, yDimensions, err := markovMutualInformationDimensions(node)
	if err != nil {
		return nil, err
	}

	outputShape, err := computetensor.NewShape([]int{1})
	if err != nil {
		return nil, err
	}

	markovOps, err := tensorBackend.markovBlanket()
	if err != nil {
		return nil, err
	}

	return markovOps.MutualInformationTensor(inputs[0], inputs[1], outputShape, xDimensions, yDimensions)
}

func (tensorBackend *TensorBackend) markovBlanket() (*MetalMarkovBlanket, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.markovOps != nil {
		return tensorBackend.markovOps, nil
	}

	markovOps, err := NewMarkovBlanket(metalLibrary(nil, "markov_blanket.metallib"))
	if err != nil {
		return nil, err
	}

	markovOps.runtime, err = tensorBackend.sharedRuntime(markovOps.runtime)

	if err != nil {
		return nil, err
	}

	tensorBackend.markovOps = markovOps

	return markovOps, nil
}

type markovPartition struct {
	outputShape   computetensor.Shape
	sensoryCount  int
	activeCount   int
	internalCount int
	externalCount int
}

func markovPartitionShape(node executor.NodeSpec, stateCount int) (markovPartition, error) {
	counts := []int{
		intConfigAny(node, -1, "sensory_count", "n_s", "Ns"),
		intConfigAny(node, -1, "active_count", "n_a", "Na"),
		intConfigAny(node, -1, "internal_count", "n_i", "Ni"),
		intConfigAny(node, -1, "external_count", "n_e", "Ne"),
	}

	if len(node.Shape) >= 5 && node.Shape[0] == stateCount {
		counts = []int{node.Shape[1], node.Shape[2], node.Shape[3], node.Shape[4]}
	}

	total := 0
	for _, count := range counts {
		if count < 0 {
			return markovPartition{}, fmt.Errorf("metal tensor: markov_blanket.partition node %q missing partition counts", node.ID)
		}

		total += count
	}

	outputShape, err := computetensor.NewShape([]int{total})
	if err != nil {
		return markovPartition{}, err
	}

	return markovPartition{
		outputShape:   outputShape,
		sensoryCount:  counts[0],
		activeCount:   counts[1],
		internalCount: counts[2],
		externalCount: counts[3],
	}, nil
}

func (tensorBackend *TensorBackend) markovPartitionMasks(
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, bool, error) {
	if len(inputs) == 2 {
		return inputs[1], false, nil
	}

	stateCount := inputs[0].Shape().Len()
	for maskIndex := 1; maskIndex < len(inputs); maskIndex++ {
		if inputs[maskIndex].Shape().Len() != stateCount {
			return nil, false, fmt.Errorf(
				"metal tensor: markov_blanket.partition node %q mask %d length mismatch",
				node.ID,
				maskIndex,
			)
		}
	}

	return tensorBackend.concatMarkovMasks(inputs[1:])
}

func (tensorBackend *TensorBackend) concatMarkovMasks(
	masks []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, bool, error) {
	shapeOps, err := tensorBackend.shape()
	if err != nil {
		return nil, false, err
	}

	current := masks[0]
	var temporary computetensor.Float64Tensor

	for maskIndex := 1; maskIndex < len(masks); maskIndex++ {
		nextShape, shapeErr := computetensor.NewShape(
			[]int{current.Shape().Len() + masks[maskIndex].Shape().Len()},
		)
		if shapeErr != nil {
			if temporary != nil {
				_ = temporary.Close()
			}

			return nil, false, shapeErr
		}

		next, concatErr := shapeOps.ConcatTensor(current, masks[maskIndex], nextShape)
		if temporary != nil {
			_ = temporary.Close()
		}
		if concatErr != nil {
			return nil, false, concatErr
		}

		current = next
		temporary = next
	}

	return current, true, nil
}

func markovFlowShape(
	node executor.NodeSpec,
	operation string,
) (computetensor.Shape, error) {
	if len(node.Shape) == 0 || node.Shape[0] <= 0 {
		return computetensor.Shape{}, fmt.Errorf("metal tensor: %s node %q has invalid shape", operation, node.ID)
	}

	return computetensor.NewShape([]int{node.Shape[0]})
}

func markovMutualInformationDimensions(node executor.NodeSpec) (int, int, error) {
	xDimensions := intConfigAny(node, -1, "x_dim", "input_dim", "N")
	yDimensions := intConfigAny(node, -1, "y_dim", "output_dim", "M")

	if len(node.Shape) >= 2 && node.Shape[0] > 0 && node.Shape[1] > 0 {
		xDimensions = node.Shape[0]
		yDimensions = node.Shape[1]
	}

	if xDimensions <= 0 || yDimensions <= 0 {
		return 0, 0, fmt.Errorf("metal tensor: markov_blanket.mutual_information dimensions are missing")
	}

	return xDimensions, yDimensions, nil
}
