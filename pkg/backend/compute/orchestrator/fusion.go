package orchestrator

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

/*
FusionOptimizer analyzes an intermediate representation graph to combine
adjacent operations into single kernels.
*/
type FusionOptimizer struct {
	capabilities Capabilities
}

/*
NewFusionOptimizer instantiates a new FusionOptimizer.
*/
func NewFusionOptimizer() *FusionOptimizer {
	return NewFusionOptimizerWithCapabilities(NewDefaultCapabilities(""))
}

func NewFusionOptimizerWithCapabilities(capabilities Capabilities) *FusionOptimizer {
	if capabilities == nil {
		capabilities = NewDefaultCapabilities("")
	}

	return &FusionOptimizer{capabilities: capabilities}
}

func (optimizer *FusionOptimizer) Name() string {
	return "fusion"
}

func (optimizer *FusionOptimizer) Run(
	ctx context.Context,
	input PassInput,
) (PassResult, error) {
	if err := ctx.Err(); err != nil {
		return PassResult{}, err
	}

	graph, targets, err := optimizer.OptimizeWithTargets(input.Graph, input.Targets)

	if err != nil {
		return PassResult{}, err
	}

	input.Diagnostics.Add(optimizer.Name(), DiagnosticInfo, "applied legal fusion patterns")

	return PassResult{
		Graph:       graph,
		Targets:     targets,
		TargetMap:   targetMap(targets),
		Diagnostics: input.Diagnostics,
		Changed:     true,
	}, nil
}

/*
Optimize traverses the graph and replaces fuseable node chains with a fused node.
*/
func (optimizer *FusionOptimizer) Optimize(graph *ir.Graph) (*ir.Graph, error) {
	optimizedGraph, _, err := optimizer.optimize(graph)

	return optimizedGraph, err
}

/*
OptimizeWithTargets returns an optimized graph and remaps requested targets
through any fused kernels that replaced them.
*/
func (optimizer *FusionOptimizer) OptimizeWithTargets(
	graph *ir.Graph,
	targets []*ir.Node,
) (*ir.Graph, []*ir.Node, error) {
	optimizedGraph, replacements, err := optimizer.optimize(graph)

	if err != nil {
		return nil, nil, err
	}

	return optimizedGraph, remapTargets(targets, replacements), nil
}

func (optimizer *FusionOptimizer) optimize(graph *ir.Graph) (*ir.Graph, map[string]*ir.Node, error) {
	if graph == nil {
		return nil, nil, fmt.Errorf("fusion optimizer: nil graph")
	}

	optimizedGraph := ir.NewGraph()

	dependents := make(map[string]int)
	for _, node := range graph.Nodes() {
		for _, in := range node.Inputs() {
			dependents[in.ID()]++
		}
	}

	fusedAway := make(map[string]string)
	activationsToFuse := make(map[string]bool)

	for _, node := range graph.Nodes() {
		isActivation := node.OpType() == ir.OpReLU || node.OpType() == ir.OpGELU

		if isActivation && len(node.Inputs()) == 1 {
			inputNode := node.Inputs()[0]
			if inputNode.OpType() == ir.OpMatmul &&
				dependents[inputNode.ID()] == 1 &&
				optimizer.capabilities.CanFuse("matmul.activation", ir.OpFused) {
				fusedAway[inputNode.ID()] = node.ID()
				activationsToFuse[node.ID()] = true
			}
		}
	}

	replacements := make(map[string]*ir.Node)

	for _, node := range graph.Nodes() {
		// Skip matmuls that are fused away
		if _, ok := fusedAway[node.ID()]; ok {
			continue
		}

		if activationsToFuse[node.ID()] {
			inputNode := node.Inputs()[0]

			fusedID := inputNode.ID() + "_fused_" + node.ID()
			fusedNode := ir.NewNode(fusedID, ir.OpFused, node.Shape())
			fusedNode.SetOperationID("fused.matmul.activation")
			fusedNode.SetValueType(node.ValueType())
			fusedNode.SetEffect(node.Effect())

			for _, in := range inputNode.Inputs() {
				if rep, ok := replacements[in.ID()]; ok {
					fusedNode.AddInput(rep)
				} else {
					fusedNode.AddInput(in)
				}
			}

			// Preserve metadata from original nodes
			for k, v := range inputNode.Metadata() {
				fusedNode.SetMetadata(k, v)
			}
			for k, v := range node.Metadata() {
				fusedNode.SetMetadata(k, v)
			}

			fusedNode.SetMetadata("base_op", string(inputNode.OpType()))
			fusedNode.SetMetadata("activation", string(node.OpType()))

			replacements[inputNode.ID()] = fusedNode
			replacements[node.ID()] = fusedNode
			optimizedGraph.AddNode(fusedNode)
			continue
		}

		newNode := ir.NewNode(node.ID(), node.OpType(), node.Shape())
		newNode.SetInPlace(node.InPlace())

		for _, in := range node.Inputs() {
			if rep, ok := replacements[in.ID()]; ok {
				newNode.AddInput(rep)
			} else {
				newNode.AddInput(in)
			}
		}
		for k, v := range node.Metadata() {
			newNode.SetMetadata(k, v)
		}

		replacements[node.ID()] = newNode
		optimizedGraph.AddNode(newNode)
	}

	return optimizedGraph, replacements, nil
}
