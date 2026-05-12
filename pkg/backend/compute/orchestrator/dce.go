package orchestrator

import (
	"context"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

/*
DCEOptimizer analyzes an intermediate representation graph to identify and
remove operations that do not contribute to the final sink nodes.
*/
type DCEOptimizer struct {
	ctx    context.Context
	cancel context.CancelFunc
	err    error
}

/*
NewDCEOptimizer instantiates a new Dead Code Elimination optimizer.
*/
func NewDCEOptimizer(ctx context.Context) *DCEOptimizer {
	ctx, cancel := context.WithCancel(ctx)

	return &DCEOptimizer{
		ctx:    ctx,
		cancel: cancel,
	}
}

/*
Optimize traverses the graph from the target sinks backwards and removes any node
that cannot be reached, creating a lean execution path.
*/
func (optimizer *DCEOptimizer) Optimize(graph *ir.Graph, targets []*ir.Node) *ir.Graph {
	if len(targets) == 0 {
		targets = graph.Sinks()
	}

	reachable := make(map[string]bool)
	var queue []*ir.Node
	queue = append(queue, targets...)

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		if !reachable[current.ID()] {
			reachable[current.ID()] = true
			queue = append(queue, current.Inputs()...)
		}
	}

	optimizedGraph := ir.NewGraph(optimizer.ctx)
	replacements := make(map[string]*ir.Node)

	for _, node := range graph.Nodes() {
		if !reachable[node.ID()] {
			continue
		}

		newNode := ir.NewNode(optimizer.ctx, node.ID(), node.OpType(), node.Shape())
		newNode.SetInPlace(node.InPlace())

		for k, v := range node.Metadata() {
			newNode.SetMetadata(k, v)
		}

		replacements[node.ID()] = newNode
		optimizedGraph.AddNode(newNode)
	}

	for _, node := range graph.Nodes() {
		if !reachable[node.ID()] {
			continue
		}

		newRep := replacements[node.ID()]
		for _, in := range node.Inputs() {
			if inRep, ok := replacements[in.ID()]; ok {
				newRep.AddInput(inRep)
			}
		}
	}

	return optimizedGraph
}
