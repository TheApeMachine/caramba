package ir

import (
	"fmt"
)

/*
Graph represents a collection of interconnected computational nodes.
It serves as the intermediate representation of the execution flow.
*/
type Graph struct {
	nodes []*Node
}

type Index struct {
	nodes map[string]*Node
	users map[string][]*Node
}

/*
NewGraph instantiates a new Graph.
It is created to abstract physical execution from the mathematical intent.
*/
func NewGraph() *Graph {
	return &Graph{
		nodes: make([]*Node, 0),
	}
}

/*
Nodes returns a defensive copy of all nodes in the graph.
*/
func (graph *Graph) Nodes() []*Node {
	out := make([]*Node, len(graph.nodes))
	copy(out, graph.nodes)
	return out
}

/*
AddNode registers a node in the computational graph.
*/
func (graph *Graph) AddNode(node *Node) {
	if node == nil {
		return
	}
	graph.nodes = append(graph.nodes, node)
}

func (graph *Graph) Index() (*Index, error) {
	nodes := graph.Nodes()
	index := &Index{
		nodes: make(map[string]*Node, len(nodes)),
		users: make(map[string][]*Node, len(nodes)),
	}

	for _, node := range nodes {
		if node.ID() == "" {
			return nil, fmt.Errorf("graph: node ID is required")
		}

		if _, ok := index.nodes[node.ID()]; ok {
			return nil, fmt.Errorf("graph: duplicate node %q", node.ID())
		}

		index.nodes[node.ID()] = node
	}

	for _, node := range nodes {
		for _, input := range node.Inputs() {
			if _, ok := index.nodes[input.ID()]; !ok {
				return nil, fmt.Errorf(
					"graph: node %q has unregistered input %q",
					node.ID(),
					input.ID(),
				)
			}

			index.users[input.ID()] = append(index.users[input.ID()], node)
		}
	}

	return index, nil
}

func (index *Index) Node(id string) *Node {
	if index == nil {
		return nil
	}

	return index.nodes[id]
}

func (index *Index) Users(id string) []*Node {
	if index == nil {
		return nil
	}

	users := index.users[id]
	out := make([]*Node, len(users))
	copy(out, users)

	return out
}

func (graph *Graph) Verify() error {
	if _, err := graph.Index(); err != nil {
		return err
	}

	if _, err := graph.TopologyLayers(); err != nil {
		return err
	}

	return nil
}

func (graph *Graph) Clone() (*Graph, map[string]*Node, error) {
	if err := graph.Verify(); err != nil {
		return nil, nil, err
	}

	clone := NewGraph()
	replacements := make(map[string]*Node, len(graph.nodes))

	for _, node := range graph.nodes {
		newNode := cloneNode(node)
		replacements[node.ID()] = newNode
		clone.AddNode(newNode)
	}

	for _, node := range graph.nodes {
		newNode := replacements[node.ID()]

		for _, input := range node.Inputs() {
			newInput, ok := replacements[input.ID()]
			if !ok {
				return nil, nil, fmt.Errorf("graph: missing clone input %q", input.ID())
			}

			newNode.AddInput(newInput)
		}
	}

	return clone, replacements, nil
}

func cloneNode(node *Node) *Node {
	newNode := NewNode(node.ID(), node.OpType(), node.Shape())
	newNode.SetOperationID(node.OperationID())
	newNode.SetValueType(node.ValueType())
	newNode.SetEffect(node.Effect())
	newNode.SetAlias(node.Alias())
	newNode.SetInPlace(node.InPlace())

	for key, value := range node.Metadata() {
		newNode.SetMetadata(key, value)
	}

	for key, value := range node.Attributes() {
		newNode.SetAttribute(key, value)
	}

	return newNode
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
func (graph *Graph) TopologyLayers() ([][]*Node, error) {
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

	processedCount := 0
	for len(currentLayer) > 0 {
		layers = append(layers, currentLayer)
		processedCount += len(currentLayer)
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

	if processedCount != len(graph.nodes) {
		return nil, fmt.Errorf("cycle detected in graph")
	}

	return layers, nil
}
