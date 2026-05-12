package ir

import (
	"context"
)

/*
Graph represents a collection of interconnected computational nodes.
It serves as the intermediate representation of the execution flow.
*/
type Graph struct {
	ctx    context.Context
	cancel context.CancelFunc
	err    error
	nodes  []*Node
}

/*
NewGraph instantiates a new Graph.
It is created to abstract physical execution from the mathematical intent.
*/
func NewGraph(ctx context.Context) *Graph {
	ctx, cancel := context.WithCancel(ctx)

	return &Graph{
		ctx:    ctx,
		cancel: cancel,
		nodes:  make([]*Node, 0),
	}
}

/*
Nodes returns all nodes in the graph.
*/
func (graph *Graph) Nodes() []*Node {
	return graph.nodes
}

/*
AddNode registers a node in the computational graph.
*/
func (graph *Graph) AddNode(node *Node) {
	graph.nodes = append(graph.nodes, node)
}

/*
Sinks returns nodes that have no dependents (i.e. output nodes).
*/
func (graph *Graph) Sinks() []*Node {
	hasDependent := make(map[string]bool)

	for _, node := range graph.nodes {
		for _, input := range node.Inputs() {
			hasDependent[input.ID()] = true
		}
	}

	var sinks []*Node

	for _, node := range graph.nodes {
		if !hasDependent[node.ID()] {
			sinks = append(sinks, node)
		}
	}

	return sinks
}

/*
TopologyLayers groups nodes into sequential execution layers.
Nodes in the same layer are completely independent of each other and can be
executed concurrently across multiple streams or command queues.
*/
func (graph *Graph) TopologyLayers() [][]*Node {
	layers := make([][]*Node, 0)

	type irNodeInfo struct {
		node       *Node
		dependents []*Node
	}

	inDegree := make(map[string]int)
	nodeMap := make(map[string]*irNodeInfo)

	for _, n := range graph.nodes {
		nodeMap[n.ID()] = &irNodeInfo{node: n, dependents: make([]*Node, 0)}
	}

	for _, n := range graph.nodes {
		inDegree[n.ID()] = len(n.Inputs())
		for _, in := range n.Inputs() {
			if info, ok := nodeMap[in.ID()]; ok {
				info.dependents = append(info.dependents, n)
			}
		}
	}

	var currentLayer []*Node
	for _, n := range graph.nodes {
		if inDegree[n.ID()] == 0 {
			currentLayer = append(currentLayer, n)
		}
	}

	for len(currentLayer) > 0 {
		layers = append(layers, currentLayer)
		var nextLayer []*Node

		for _, n := range currentLayer {
			for _, dep := range nodeMap[n.ID()].dependents {
				inDegree[dep.ID()]--
				if inDegree[dep.ID()] == 0 {
					nextLayer = append(nextLayer, dep)
				}
			}
		}

		currentLayer = nextLayer
	}

	return layers
}
