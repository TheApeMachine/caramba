//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (tensorBackend *TensorBackend) applyHawkesIntensity(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Tensor,
) (computetensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 5 {
		return nil, fmt.Errorf("metal tensor: hawkes.intensity node %q requires 5 inputs", node.ID)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	hawkesOps, err := tensorBackend.hawkes()
	if err != nil {
		return nil, err
	}

	return hawkesOps.IntensityTensor(
		inputs[0],
		inputs[1],
		inputs[2],
		inputs[3],
		inputs[4],
		outputShape,
	)
}

func (tensorBackend *TensorBackend) applyHawkesKernelMatrix(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Tensor,
) (computetensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("metal tensor: hawkes.kernel_matrix node %q requires 3 inputs", node.ID)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	hawkesOps, err := tensorBackend.hawkes()
	if err != nil {
		return nil, err
	}

	return hawkesOps.KernelMatrixTensor(inputs[0], inputs[1], inputs[2], outputShape)
}

func (tensorBackend *TensorBackend) applyHawkesLogLikelihood(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Tensor,
) (computetensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("metal tensor: hawkes.log_likelihood node %q requires 3 inputs", node.ID)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	hawkesOps, err := tensorBackend.hawkes()
	if err != nil {
		return nil, err
	}

	return hawkesOps.LogLikelihoodTensor(inputs[1], inputs[2], outputShape)
}

func (tensorBackend *TensorBackend) applyHawkesSimulate(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Tensor,
) (computetensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 4 {
		return nil, fmt.Errorf("metal tensor: hawkes.simulate node %q requires 4 inputs", node.ID)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	hawkesOps, err := tensorBackend.hawkes()
	if err != nil {
		return nil, err
	}

	return hawkesOps.SimulateTensor(inputs[0], inputs[1], inputs[2], inputs[3], outputShape)
}

func (tensorBackend *TensorBackend) hawkes() (*MetalHawkes, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.hawkesOps != nil {
		return tensorBackend.hawkesOps, nil
	}

	hawkesOps, err := NewHawkes(metalLibrary(nil, "hawkes.metallib"))
	if err != nil {
		return nil, err
	}

	hawkesOps.runtime, err = tensorBackend.sharedRuntime(hawkesOps.runtime)

	if err != nil {
		return nil, err
	}

	tensorBackend.hawkesOps = hawkesOps

	return hawkesOps, nil
}
