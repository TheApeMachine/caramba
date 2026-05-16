//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (tensorBackend *TensorBackend) applyCausalCounterfactual(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 4 {
		return nil, fmt.Errorf("metal tensor: causal.counterfactual node %q requires 4 inputs", node.ID)
	}

	outputShape, err := causalCounterfactualOutputShape(node, inputs[0], inputs[3])
	if err != nil {
		return nil, err
	}

	causalOps, err := tensorBackend.causal()
	if err != nil {
		return nil, err
	}

	return causalOps.CounterfactualTensor(inputs[0], inputs[1], inputs[2], inputs[3], outputShape)
}

func (tensorBackend *TensorBackend) applyCausalFrontdoorAdjustment(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("metal tensor: causal.frontdoor_adjustment node %q requires 3 inputs", node.ID)
	}

	config, err := causalFrontdoorConfig(node, inputs[0])
	if err != nil {
		return nil, err
	}

	causalOps, err := tensorBackend.causal()
	if err != nil {
		return nil, err
	}

	return causalOps.FrontdoorAdjustmentTensor(
		inputs[0],
		inputs[1],
		inputs[2],
		config.outputShape,
		config.samples,
		config.treatmentBins,
		config.mediatorBins,
	)
}

func (tensorBackend *TensorBackend) causal() (*MetalCausalOps, error) {
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	if tensorBackend.causalOps != nil {
		return tensorBackend.causalOps, nil
	}

	causalOps, err := NewCausalOps(metalLibrary(nil, "causal.metallib"))
	if err != nil {
		return nil, err
	}

	causalOps.runtime = tensorBackend.runtime
	tensorBackend.causalOps = causalOps

	return causalOps, nil
}

func causalCounterfactualOutputShape(
	node executor.NodeSpec,
	observed computetensor.Float64Tensor,
	counterfactual computetensor.Float64Tensor,
) (computetensor.Shape, error) {
	observedCount := observed.Shape().Len()
	counterfactualCount := counterfactual.Shape().Len()
	if observedCount <= 0 || counterfactualCount <= 0 {
		return computetensor.Shape{}, fmt.Errorf("metal tensor: causal.counterfactual node %q has invalid input shape", node.ID)
	}

	expectedElements := observedCount * counterfactualCount
	if len(node.Shape) >= 2 && node.Shape[0] == observedCount && node.Shape[1] == counterfactualCount {
		return computetensor.NewShape([]int{observedCount, counterfactualCount})
	}

	if len(node.Shape) == 1 && node.Shape[0] == expectedElements {
		return computetensor.NewShape(node.Shape)
	}

	return computetensor.NewShape([]int{observedCount, counterfactualCount})
}

type causalFrontdoor struct {
	outputShape   computetensor.Shape
	samples       int
	treatmentBins int
	mediatorBins  int
}

func causalFrontdoorConfig(
	node executor.NodeSpec,
	treatment computetensor.Float64Tensor,
) (causalFrontdoor, error) {
	treatmentBins := intConfigAny(node, -1, "treatment_bins", "x_bins", "N_x")
	mediatorBins := intConfigAny(node, -1, "mediator_bins", "m_bins", "N_m")
	samples := intConfigAny(node, treatment.Shape().Len(), "samples", "T")

	if len(node.Shape) >= 4 {
		treatmentBins = node.Shape[0]
		mediatorBins = node.Shape[1]
		samples = node.Shape[3]
	}

	if treatmentBins <= 0 || mediatorBins <= 0 || samples <= 0 {
		return causalFrontdoor{}, fmt.Errorf("metal tensor: causal.frontdoor_adjustment node %q has invalid shape", node.ID)
	}

	outputShape, err := computetensor.NewShape([]int{treatmentBins})
	if err != nil {
		return causalFrontdoor{}, err
	}

	return causalFrontdoor{
		outputShape:   outputShape,
		samples:       samples,
		treatmentBins: treatmentBins,
		mediatorBins:  mediatorBins,
	}, nil
}
