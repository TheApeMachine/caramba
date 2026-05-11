package manifest

import (
	"fmt"
	"maps"

	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation"
)

/*
Node is a single operation in the computation graph.
It holds the resolved operation and its wiring metadata.
*/
type Node struct {
	ID  string
	Op  operation.Operation
	In  []string
	Out []string
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
	nodes []*Node
	edges []*Edge
	index map[string]*Node
}

func newGraph() *Graph {
	return &Graph{index: make(map[string]*Node)}
}

func (graph *Graph) addNode(node *Node) {
	graph.nodes = append(graph.nodes, node)
	graph.index[node.ID] = node
}

func (graph *Graph) addEdge(edge *Edge) {
	graph.edges = append(graph.edges, edge)
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
				continue
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
	state := make(map[string][]float64, len(inputs))

	maps.Copy(state, inputs)

	for _, node := range graph.nodes {
		data, err := graph.gatherInputsForNode(node, state)

		if err != nil {
			return nil, err
		}

		output := node.Op.Forward(shape, data...)

		if len(node.Out) == 0 {
			state[node.ID] = output

			continue
		}

		state[node.Out[0]] = output
	}

	return state, nil
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
