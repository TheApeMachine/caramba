package manifest

import (
	"fmt"
	"maps"

	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Node is a single operation in the computation graph.
It holds the resolved operation and its wiring metadata.
*/
type Node struct {
	ID     string
	OpID   string
	Config map[string]any
	Op     operation.Operation
	In     []string
	Out    []string
}

/*
Edge connects one node's output port to another node's input port.
*/
type Edge struct {
	From     string
	FromPort string
	To       string
	ToPort   string
}

/*
Graph is an ordered, executable computation graph built from a topology manifest.
Nodes are stored in topological order so Execute can run them left-to-right.
*/
type Graph struct {
	nodes          []*Node
	edges          []*Edge
	index          map[string]*Node
	externalInputs map[string]bool
}

func newGraph() *Graph {
	return &Graph{
		index:          make(map[string]*Node),
		externalInputs: make(map[string]bool),
	}
}

func (graph *Graph) addNode(node *Node) error {
	if node == nil {
		return fmt.Errorf("graph: nil node")
	}

	if node.ID == "" {
		return fmt.Errorf("graph: node id is required")
	}

	if _, exists := graph.index[node.ID]; exists {
		return fmt.Errorf("graph: duplicate node id %q", node.ID)
	}

	if len(node.Out) > 1 {
		return fmt.Errorf("graph: node %q declares %d outputs; multi-output operations are not supported", node.ID, len(node.Out))
	}

	graph.nodes = append(graph.nodes, node)
	graph.index[node.ID] = node

	return nil
}

func (graph *Graph) addEdge(edge *Edge) {
	graph.edges = append(graph.edges, edge)
}

func (graph *Graph) Nodes() []*Node {
	nodes := make([]*Node, len(graph.nodes))
	copy(nodes, graph.nodes)

	return nodes
}

func (graph *Graph) Edges() []*Edge {
	edges := make([]*Edge, len(graph.edges))
	copy(edges, graph.edges)

	return edges
}

/*
publicationBinding is the state key this node writes to after Forward, matching Execute.
*/
func publicationBinding(node *Node) string {
	if len(node.Out) == 0 {
		return node.ID
	}

	return node.Out[0]
}

/*
rebuildEdgesFromNodes clears and repopulates graph.edges from node in/out bindings.
Each edge links the unique producer of a binding to a consumer that lists that binding in In.
Graph inputs (no producer) produce no edge; duplicate producers for the same binding error.
*/
func (graph *Graph) rebuildEdgesFromNodes() error {
	graph.edges = graph.edges[:0]

	producerByBinding := make(map[string]string)

	for _, producer := range graph.nodes {
		binding := publicationBinding(producer)

		if priorID, exists := producerByBinding[binding]; exists {
			return fmt.Errorf("graph: duplicate publication of binding %q from nodes %q and %q", binding, priorID, producer.ID)
		}

		producerByBinding[binding] = producer.ID
	}

	for _, consumer := range graph.nodes {
		for _, inputBinding := range consumer.In {
			producerID, found := producerByBinding[inputBinding]

			if !found {
				if graph.externalInputs[inputBinding] {
					continue
				}

				return fmt.Errorf("graph: node %q input %q has no producer or declared external input", consumer.ID, inputBinding)
			}

			if producerID == consumer.ID {
				continue
			}

			graph.addEdge(&Edge{
				From:     producerID,
				FromPort: inputBinding,
				To:       consumer.ID,
				ToPort:   inputBinding,
			})
		}
	}

	return nil
}

/*
Execute runs the graph with named inputs,
propagates outputs along edges, and returns accumulated state (including intermediates).
*/
func (graph *Graph) Execute(inputs map[string][]float64, shape []int) (map[string][]float64, error) {
	stateMap := make(map[string][]float64, len(inputs))

	maps.Copy(stateMap, inputs)

	order, err := graph.executionOrder()

	if err != nil {
		return nil, err
	}

	for _, node := range order {
		data, err := graph.gatherInputsForNode(node, stateMap)

		if err != nil {
			return nil, err
		}

		stateDict := state.NewDict().WithShape(shape)
		stateDict.Inputs = data

		outputState, err := node.Op.Forward(stateDict)

		if err != nil {
			return nil, err
		}

		if len(node.Out) == 0 {
			stateMap[node.ID] = outputState.Out

			continue
		}

		if len(node.Out) != 1 {
			return nil, fmt.Errorf("graph: node %q must publish exactly one output", node.ID)
		}

		stateMap[node.Out[0]] = outputState.Out
	}

	return stateMap, nil
}

func (graph *Graph) executionOrder() ([]*Node, error) {
	indegree := make(map[string]int, len(graph.nodes))
	dependents := make(map[string][]string, len(graph.nodes))

	for _, node := range graph.nodes {
		indegree[node.ID] = 0
	}

	for _, edge := range graph.edges {
		indegree[edge.To]++
		dependents[edge.From] = append(dependents[edge.From], edge.To)
	}

	queue := make([]*Node, 0, len(graph.nodes))

	for _, node := range graph.nodes {
		if indegree[node.ID] == 0 {
			queue = append(queue, node)
		}
	}

	ordered := make([]*Node, 0, len(graph.nodes))

	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		ordered = append(ordered, node)

		for _, dependentID := range dependents[node.ID] {
			indegree[dependentID]--

			if indegree[dependentID] == 0 {
				queue = append(queue, graph.index[dependentID])
			}
		}
	}

	if len(ordered) != len(graph.nodes) {
		return nil, fmt.Errorf("graph: cycle detected")
	}

	return ordered, nil
}

/*
gatherInputsForNode collects ordered input buffers for node from state.
*/
func (graph *Graph) gatherInputsForNode(node *Node, state map[string][]float64) ([][]float64, error) {
	data := make([][]float64, 0, len(node.In))

	for _, portBinding := range node.In {
		buffer, ok := state[portBinding]

		if !ok {
			return nil, fmt.Errorf("graph: node %q input %q not in state", node.ID, portBinding)
		}

		data = append(data, buffer)
	}

	return data, nil
}

/*
Weights returns the current parameter state of each node, keyed by node ID.
Nodes whose Operation does not implement Parameterized are skipped.
*/
func (graph *Graph) Weights() (map[string][]float64, error) {
	out := make(map[string][]float64, len(graph.nodes))

	for _, node := range graph.nodes {
		parameterized, ok := node.Op.(operation.Parameterized)

		if !ok {
			continue
		}

		out[node.ID] = parameterized.Params()
	}

	return out, nil
}

/*
LoadWeights restores per-node parameters from a previously saved state map.
Nodes absent from state are left untouched.
*/
func (graph *Graph) LoadWeights(state map[string][]float64) error {
	for nodeID, params := range state {
		node, ok := graph.index[nodeID]

		if !ok {
			continue
		}

		parameterized, ok := node.Op.(operation.Parameterized)

		if !ok {
			return fmt.Errorf("graph: node %q is not parameterized", nodeID)
		}

		parameterized.SetParams(params)
	}

	return nil
}
