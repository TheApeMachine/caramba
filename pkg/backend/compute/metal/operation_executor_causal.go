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

func (tensorBackend *TensorBackend) applyCausalBackdoorAdjustment(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("metal tensor: causal.backdoor_adjustment node %q requires 3 inputs", node.ID)
	}

	config, err := causalBackdoorConfig(node)
	if err != nil {
		return nil, err
	}

	causalOps, err := tensorBackend.causal()
	if err != nil {
		return nil, err
	}

	return causalOps.BackdoorAdjustmentTensor(
		inputs[0],
		inputs[1],
		inputs[2],
		config.outputShape,
		config.samples,
		config.outcomeDimensions,
		config.treatmentDimensions,
		config.confounderDimensions,
	)
}

func (tensorBackend *TensorBackend) applyCausalCATE(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("metal tensor: causal.cate node %q requires 3 inputs", node.ID)
	}

	config, err := causalCATEConfig(node)
	if err != nil {
		return nil, err
	}

	causalOps, err := tensorBackend.causal()
	if err != nil {
		return nil, err
	}

	return causalOps.CATETensor(
		inputs[0],
		inputs[1],
		inputs[2],
		config.outputShape,
		config.samples,
		config.covariateDimensions,
	)
}

func (tensorBackend *TensorBackend) applyCausalIVEstimate(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("metal tensor: causal.iv_estimate node %q requires 3 inputs", node.ID)
	}

	config, err := causalIVConfig(node)
	if err != nil {
		return nil, err
	}

	causalOps, err := tensorBackend.causal()
	if err != nil {
		return nil, err
	}

	return causalOps.IVEstimateTensor(
		inputs[0],
		inputs[1],
		inputs[2],
		config.outputShape,
		config.samples,
		config.instrumentDimensions,
		config.treatmentDimensions,
		config.outcomeDimensions,
	)
}

func (tensorBackend *TensorBackend) applyCausalDAGMarkovFactorization(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 2 {
		return nil, fmt.Errorf("metal tensor: causal.dag_markov_factorization node %q requires 2 inputs", node.ID)
	}

	config, err := causalDAGConfig(node)
	if err != nil {
		return nil, err
	}

	causalOps, err := tensorBackend.causal()
	if err != nil {
		return nil, err
	}

	return causalOps.DAGMarkovFactorizationTensor(
		inputs[0],
		inputs[1],
		config.outputShape,
		config.nodeCount,
		config.samples,
	)
}

func (tensorBackend *TensorBackend) applyCausalDoCalculus(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 3 {
		return nil, fmt.Errorf("metal tensor: causal.do_calculus node %q requires 3 inputs", node.ID)
	}

	config, err := causalDoConfig(node)
	if err != nil {
		return nil, err
	}

	causalOps, err := tensorBackend.causal()
	if err != nil {
		return nil, err
	}

	return causalOps.DoCalculusTensor(
		inputs[0],
		inputs[1],
		inputs[2],
		config.outputShape,
		config.nodeCount,
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

type causalDo struct {
	outputShape computetensor.Shape
	nodeCount   int
}

func causalDoConfig(node executor.NodeSpec) (causalDo, error) {
	nodeCount := intConfigAny(node, -1, "node_count", "N")

	if len(node.Shape) >= 1 {
		nodeCount = node.Shape[0]
	}

	if nodeCount <= 0 {
		return causalDo{}, fmt.Errorf("metal tensor: causal.do_calculus node %q has invalid shape", node.ID)
	}

	outputShape, err := computetensor.NewShape([]int{nodeCount + nodeCount*nodeCount})
	if err != nil {
		return causalDo{}, err
	}

	return causalDo{
		outputShape: outputShape,
		nodeCount:   nodeCount,
	}, nil
}

type causalDAG struct {
	outputShape computetensor.Shape
	nodeCount   int
	samples     int
}

func causalDAGConfig(node executor.NodeSpec) (causalDAG, error) {
	nodeCount := intConfigAny(node, -1, "node_count", "N")
	samples := intConfigAny(node, -1, "samples", "T")

	if len(node.Shape) >= 2 {
		nodeCount = node.Shape[0]
		samples = node.Shape[1]
	}

	if nodeCount <= 0 || samples <= 0 {
		return causalDAG{}, fmt.Errorf("metal tensor: causal.dag_markov_factorization node %q has invalid shape", node.ID)
	}

	outputShape, err := computetensor.NewShape([]int{samples})
	if err != nil {
		return causalDAG{}, err
	}

	return causalDAG{
		outputShape: outputShape,
		nodeCount:   nodeCount,
		samples:     samples,
	}, nil
}

type causalIV struct {
	outputShape          computetensor.Shape
	samples              int
	instrumentDimensions int
	treatmentDimensions  int
	outcomeDimensions    int
}

func causalIVConfig(node executor.NodeSpec) (causalIV, error) {
	samples := intConfigAny(node, -1, "samples", "T")
	instrumentDimensions := intConfigAny(node, -1, "instrument_dims", "N_z")
	treatmentDimensions := intConfigAny(node, -1, "treatment_dims", "N_x")
	outcomeDimensions := intConfigAny(node, -1, "outcome_dims", "N_y")

	if len(node.Shape) >= 4 {
		samples = node.Shape[0]
		instrumentDimensions = node.Shape[1]
		treatmentDimensions = node.Shape[2]
		outcomeDimensions = node.Shape[3]
	}

	if samples <= 0 || instrumentDimensions <= 0 || treatmentDimensions <= 0 || outcomeDimensions <= 0 {
		return causalIV{}, fmt.Errorf("metal tensor: causal.iv_estimate node %q has invalid shape", node.ID)
	}

	outputShape, err := computetensor.NewShape([]int{treatmentDimensions, outcomeDimensions})
	if err != nil {
		return causalIV{}, err
	}

	return causalIV{
		outputShape:          outputShape,
		samples:              samples,
		instrumentDimensions: instrumentDimensions,
		treatmentDimensions:  treatmentDimensions,
		outcomeDimensions:    outcomeDimensions,
	}, nil
}

type causalCATE struct {
	outputShape         computetensor.Shape
	samples             int
	covariateDimensions int
}

func causalCATEConfig(node executor.NodeSpec) (causalCATE, error) {
	samples := intConfigAny(node, -1, "samples", "T")
	covariateDimensions := intConfigAny(node, -1, "covariate_dims", "N_x")

	if len(node.Shape) >= 2 {
		samples = node.Shape[0]
		covariateDimensions = node.Shape[1]
	}

	if samples <= 0 || covariateDimensions <= 0 {
		return causalCATE{}, fmt.Errorf("metal tensor: causal.cate node %q has invalid shape", node.ID)
	}

	outputShape, err := computetensor.NewShape([]int{samples})
	if err != nil {
		return causalCATE{}, err
	}

	return causalCATE{
		outputShape:         outputShape,
		samples:             samples,
		covariateDimensions: covariateDimensions,
	}, nil
}

type causalBackdoor struct {
	outputShape          computetensor.Shape
	samples              int
	outcomeDimensions    int
	treatmentDimensions  int
	confounderDimensions int
}

func causalBackdoorConfig(node executor.NodeSpec) (causalBackdoor, error) {
	outcomeDimensions := intConfigAny(node, -1, "outcome_dims", "N_y")
	treatmentDimensions := intConfigAny(node, -1, "treatment_dims", "N_x")
	confounderDimensions := intConfigAny(node, -1, "confounder_dims", "N_z")
	samples := intConfigAny(node, -1, "samples", "T")

	if len(node.Shape) >= 4 {
		outcomeDimensions = node.Shape[0]
		treatmentDimensions = node.Shape[1]
		confounderDimensions = node.Shape[2]
		samples = node.Shape[3]
	}

	if outcomeDimensions <= 0 || treatmentDimensions <= 0 || confounderDimensions < 0 || samples <= 0 {
		return causalBackdoor{}, fmt.Errorf("metal tensor: causal.backdoor_adjustment node %q has invalid shape", node.ID)
	}

	outputShape, err := computetensor.NewShape([]int{outcomeDimensions})
	if err != nil {
		return causalBackdoor{}, err
	}

	return causalBackdoor{
		outputShape:          outputShape,
		samples:              samples,
		outcomeDimensions:    outcomeDimensions,
		treatmentDimensions:  treatmentDimensions,
		confounderDimensions: confounderDimensions,
	}, nil
}
