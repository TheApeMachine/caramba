package orchestrator

import (
	"context"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

/*
FusionOptimizer analyzes an intermediate representation graph to combine
adjacent operations into single kernels.
*/
type FusionOptimizer struct {
	ctx    context.Context
	cancel context.CancelFunc
	err    error
}

/*
NewFusionOptimizer instantiates a new FusionOptimizer.
*/
func NewFusionOptimizer(ctx context.Context) *FusionOptimizer {
	ctx, cancel := context.WithCancel(ctx)

	return &FusionOptimizer{
		ctx:    ctx,
		cancel: cancel,
	}
}

/*
Optimize traverses the graph and replaces fuseable node chains with a fused node.
*/
func (optimizer *FusionOptimizer) Optimize(graph *ir.Graph) *ir.Graph {
	optimizedGraph := ir.NewGraph(optimizer.ctx)

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
			if inputNode.OpType() == ir.OpMatmul && dependents[inputNode.ID()] == 1 {
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
			fusedNode := ir.NewNode(optimizer.ctx, fusedID, ir.OpFused, node.Shape())

			for _, in := range inputNode.Inputs() {
				if rep, ok := replacements[in.ID()]; ok {
					fusedNode.AddInput(rep)
				} else {
					fusedNode.AddInput(in)
				}
			}

			fusedNode.SetMetadata("base_op", string(inputNode.OpType()))
			fusedNode.SetMetadata("activation", string(node.OpType()))

			replacements[node.ID()] = fusedNode
			optimizedGraph.AddNode(fusedNode)
			continue
		}

		newNode := ir.NewNode(optimizer.ctx, node.ID(), node.OpType(), node.Shape())
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

	return optimizedGraph
}
