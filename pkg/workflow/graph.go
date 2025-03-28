package workflow

import (
	"errors"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type Node struct {
	ID        string
	Component io.ReadWriter
}

type Edge struct {
	From string
	To   string
}

type Graph struct {
	registry  io.ReadWriteCloser
	nodes     map[string]*Node
	edges     map[string][]string
	processed bool
}

type GraphOption func(*Graph)

func NewGraph(opts ...GraphOption) *Graph {
	errnie.Debug("graph.NewGraph")

	graph := &Graph{
		registry: nil,
		nodes:    make(map[string]*Node),
		edges:    make(map[string][]string),
	}

	for _, opt := range opts {
		opt(graph)
	}

	return graph
}

func (graph *Graph) Read(p []byte) (n int, err error) {
	errnie.Debug("graph.Read")

	if graph.registry == nil {
		return 0, errnie.Error(errors.New("registry not set"))
	}

	if !graph.processed {
		// Process through all edges in the graph
		for from, targets := range graph.edges {
			if node, ok := graph.nodes[from]; ok {
				errnie.Info("graph.Read", "from", node.ID)
				// Copy from source node to all target nodes
				for _, to := range targets {
					if targetNode, ok := graph.nodes[to]; ok {
						errnie.Info("graph.Read", "to", targetNode.ID)
						if _, err = io.Copy(targetNode.Component, node.Component); err != nil {
							return 0, errnie.Error(err)
						}
					}
				}
			}
		}

		graph.processed = true
	}

	// Read from registry after processing
	return graph.registry.Read(p)
}

func (graph *Graph) Write(p []byte) (n int, err error) {
	errnie.Debug("graph.Write")

	if graph.registry == nil {
		return 0, errnie.Error(errors.New("registry not set"))
	}

	// Reset processed state when new data comes in
	graph.processed = false

	// Write to registry first
	if n, err = graph.registry.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	return n, nil
}

func (graph *Graph) Close() error {
	errnie.Debug("graph.Close")

	if graph.registry == nil {
		return errnie.Error(errors.New("registry not set"))
	}

	return graph.registry.Close()
}

func WithRegistry(registry io.ReadWriteCloser) GraphOption {
	return func(graph *Graph) {
		graph.registry = registry
	}
}

func WithNode(node *Node) GraphOption {
	return func(graph *Graph) {
		graph.nodes[node.ID] = node
	}
}

func WithEdge(edge *Edge) GraphOption {
	return func(graph *Graph) {
		graph.edges[edge.From] = append(graph.edges[edge.From], edge.To)
	}
}
