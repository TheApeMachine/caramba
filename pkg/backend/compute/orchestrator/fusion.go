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
	if err := checkContext(ctx); err != nil {
		return PassResult{}, err
	}

	graph, targets, err := optimizer.OptimizeWithTargets(ctx, input.Graph, input.Targets)

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
	optimizedGraph, _, err := optimizer.optimize(context.Background(), graph)

	return optimizedGraph, err
}

/*
OptimizeWithTargets returns an optimized graph and remaps requested targets
through any fused kernels that replaced them.
*/
func (optimizer *FusionOptimizer) OptimizeWithTargets(
	ctx context.Context,
	graph *ir.Graph,
	targets []*ir.Node,
) (*ir.Graph, []*ir.Node, error) {
	optimizedGraph, replacements, err := optimizer.optimize(ctx, graph)

	if err != nil {
		return nil, nil, err
	}

	return optimizedGraph, remapTargets(targets, replacements), nil
}

func (optimizer *FusionOptimizer) optimize(ctx context.Context, graph *ir.Graph) (*ir.Graph, map[string]*ir.Node, error) {
	if graph == nil {
		return nil, nil, fmt.Errorf("fusion optimizer: nil graph")
	}

	if err := checkContext(ctx); err != nil {
		return nil, nil, err
	}

	optimizedGraph := ir.NewGraph()

	dependents := make(map[string]int)
	for _, node := range graph.Nodes() {
		if err := checkContext(ctx); err != nil {
			return nil, nil, err
		}

		for _, in := range node.Inputs() {
			dependents[in.ID()]++
		}
	}

	fusedAway := make(map[string]string)
	activationsToFuse := make(map[string]bool)

	for _, node := range graph.Nodes() {
		if err := checkContext(ctx); err != nil {
			return nil, nil, err
		}

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
	fusedNodes := make(map[string]*ir.Node)

	for _, node := range graph.Nodes() {
		if err := checkContext(ctx); err != nil {
			return nil, nil, err
		}

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
			fusedNodes[node.ID()] = fusedNode
			optimizedGraph.AddNode(fusedNode)
			continue
		}

		newNode := cloneNodeSemantics(node)
		replacements[node.ID()] = newNode
		optimizedGraph.AddNode(newNode)
	}

	for _, node := range graph.Nodes() {
		if err := checkContext(ctx); err != nil {
			return nil, nil, err
		}

		if _, ok := fusedAway[node.ID()]; ok {
			continue
		}

		newNode := replacements[node.ID()]
		inputNode := node

		if fusedNode, ok := fusedNodes[node.ID()]; ok {
			newNode = fusedNode
			inputNode = node.Inputs()[0]
		}

		for _, in := range inputNode.Inputs() {
			newNode.AddInput(replacements[in.ID()])
		}
	}

	return optimizedGraph, replacements, nil
}
