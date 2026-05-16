//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (tensorBackend *TensorBackend) applyVSABind(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 2 {
		return nil, fmt.Errorf("metal tensor: vsa.bind node %q requires 2 inputs", node.ID)
	}

	vsaOps, err := tensorBackend.vsa()
	if err != nil {
		return nil, err
	}

	return vsaOps.BindTensor(inputs[0], inputs[1])
}

func (tensorBackend *TensorBackend) applyVSABundle(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) == 0 {
		return nil, fmt.Errorf("metal tensor: vsa.bundle node %q requires input tensors", node.ID)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	vsaOps, err := tensorBackend.vsa()
	if err != nil {
		return nil, err
	}

	if len(inputs) == 1 {
		count, countErr := vsaBundleCount(node, inputs[0], outputShape)
		if countErr != nil {
			return nil, countErr
		}

		return vsaOps.BundleTensor(inputs[0], outputShape, count)
	}

	vectors, err := tensorBackend.vsaBundleVectors(node, inputs, outputShape)
	if err != nil {
		return nil, err
	}
	defer vectors.Close()

	return vsaOps.BundleTensor(vectors, outputShape, len(inputs))
}

func (tensorBackend *TensorBackend) applyVSASimilarity(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 2 {
		return nil, fmt.Errorf("metal tensor: vsa.similarity node %q requires 2 inputs", node.ID)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	vsaOps, err := tensorBackend.vsa()
	if err != nil {
		return nil, err
	}

	return vsaOps.SimilarityTensor(inputs[0], inputs[1], outputShape)
}

func (tensorBackend *TensorBackend) applyVSAPermute(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.applyVSAPermutation(ctx, node, inputs, false)
}

func (tensorBackend *TensorBackend) applyVSAInversePermute(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return tensorBackend.applyVSAPermutation(ctx, node, inputs, true)
}

func (tensorBackend *TensorBackend) applyVSAPermutation(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
	inverse bool,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: %s node %q requires 1 input", node.Op, node.ID)
	}

	vsaOps, err := tensorBackend.vsa()
	if err != nil {
		return nil, err
	}

	shift := vsaShift(node, 1)
	if inverse {
		return vsaOps.InversePermuteTensor(inputs[0], shift)
	}

	return vsaOps.PermuteTensor(inputs[0], shift)
}

func (tensorBackend *TensorBackend) vsaBundleVectors(
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
	outputShape computetensor.Shape,
) (computetensor.Float64Tensor, error) {
	for inputIndex, input := range inputs {
		if input.Shape().Len() != outputShape.Len() {
			return nil, fmt.Errorf(
				"metal tensor: vsa.bundle node %q input %d length mismatch",
				node.ID,
				inputIndex,
			)
		}
	}

	shapeOps, err := tensorBackend.shape()
	if err != nil {
		return nil, err
	}

	current := inputs[0]
	var temporary computetensor.Float64Tensor

	for inputIndex := 1; inputIndex < len(inputs); inputIndex++ {
		nextShape, shapeErr := computetensor.NewShape(
			[]int{current.Shape().Len() + inputs[inputIndex].Shape().Len()},
		)
		if shapeErr != nil {
			if temporary != nil {
				_ = temporary.Close()
			}

			return nil, shapeErr
		}

		next, concatErr := shapeOps.ConcatTensor(current, inputs[inputIndex], nextShape)
		if temporary != nil {
			_ = temporary.Close()
		}

		if concatErr != nil {
			return nil, concatErr
		}

		current = next
		temporary = next
	}

	return current, nil
}

func (tensorBackend *TensorBackend) vsa() (*MetalVSAOps, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.vsaOps != nil {
		return tensorBackend.vsaOps, nil
	}

	vsaOps, err := NewVSAOps(metalLibrary(nil, "vsa.metallib"))
	if err != nil {
		return nil, err
	}

	vsaOps.runtime = tensorBackend.runtime
	tensorBackend.vsaOps = vsaOps

	return vsaOps, nil
}

func vsaBundleCount(
	node executor.NodeSpec,
	input computetensor.Float64Tensor,
	outputShape computetensor.Shape,
) (int, error) {
	if outputShape.Len() <= 0 {
		return 0, fmt.Errorf("metal tensor: vsa.bundle node %q output shape is empty", node.ID)
	}

	count := intConfig(node, "count", 0)
	if count == 0 {
		count = intConfig(node, "num_vectors", 0)
	}

	if count == 0 && input.Shape().Len()%outputShape.Len() == 0 {
		count = input.Shape().Len() / outputShape.Len()
	}

	if count <= 0 || input.Shape().Len() != count*outputShape.Len() {
		return 0, fmt.Errorf("metal tensor: vsa.bundle node %q shape mismatch", node.ID)
	}

	return count, nil
}

func vsaShift(node executor.NodeSpec, defaultShift int) int {
	value, ok := node.Metadata["k"]
	if ok {
		return intValue(value, defaultShift)
	}

	value, ok = node.Metadata["shift"]
	if ok {
		return intValue(value, defaultShift)
	}

	return defaultShift
}

func intValue(value any, defaultValue int) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case float32:
		return int(typed)
	default:
		return defaultValue
	}
}
