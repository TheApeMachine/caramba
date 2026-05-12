package orchestrator

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

/*
DCEOptimizer analyzes an intermediate representation graph to identify and
remove operations that do not contribute to the final sink nodes.
*/
type DCEOptimizer struct {
}

/*
NewDCEOptimizer instantiates a new Dead Code Elimination optimizer.
*/
func NewDCEOptimizer() *DCEOptimizer {
	return &DCEOptimizer{}
}

/*
Optimize traverses the graph from the target sinks backwards and removes any node
that cannot be reached, creating a lean execution path.
*/
func (optimizer *DCEOptimizer) Optimize(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (*ir.Graph, error) {
	if graph == nil {
		return nil, fmt.Errorf("dce optimizer: nil graph")
	}

	if len(targets) == 0 {
		targets = graph.Sinks()
	}

	reachable := make(map[string]bool)
	var queue []*ir.Node
	queue = append(queue, targets...)

	head := 0
	for head < len(queue) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		current := queue[head]
		head++

		if !reachable[current.ID()] {
			reachable[current.ID()] = true
			queue = append(queue, current.Inputs()...)
		}
	}

	optimizedGraph := ir.NewGraph()
	replacements := make(map[string]*ir.Node)

	for _, node := range graph.Nodes() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		if !reachable[node.ID()] {
			continue
		}

		newNode := ir.NewNode(node.ID(), node.OpType(), node.Shape())
		newNode.SetInPlace(node.InPlace())

		for k, v := range node.Metadata() {
			newNode.SetMetadata(k, v)
		}

		replacements[node.ID()] = newNode
		optimizedGraph.AddNode(newNode)
	}

	for _, node := range graph.Nodes() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

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

	return optimizedGraph, nil
}
