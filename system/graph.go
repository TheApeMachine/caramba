package system

import (
	"context"

	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

type Graph struct {
	Nodes []*Node `json:"nodes" jsonschema:"title=Nodes,description=The nodes of the graph,required"`
	Edges []*Edge `json:"edges" jsonschema:"title=Edges,description=The edges of the graph,required"`
}

func NewGraph(nodes []*Node, edges []*Edge) *Graph {
	return &Graph{
		Nodes: nodes,
		Edges: edges,
	}
}

func (graph *Graph) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Graph]()
}

func (graph *Graph) Generate(ctx context.Context, message *provider.Message) <-chan provider.Event {
	out := make(chan provider.Event)

	// Create a map of nodes for easy lookup
	nodeMap := make(map[string]*Node)
	for _, node := range graph.Nodes {
		nodeMap[node.ID] = node
	}

	// Create a map of outgoing edges for each node
	edgeMap := make(map[string][]*Edge)
	for _, edge := range graph.Edges {
		edgeMap[edge.From] = append(edgeMap[edge.From], edge)
	}

	go func() {
		defer close(out)

		// Find root nodes (nodes with no incoming edges)
		rootNodes := graph.findRootNodes()

		// Process each root node
		for _, rootNode := range rootNodes {
			if rootNode.Parallel {
				go graph.processNode(ctx, rootNode, nodeMap, edgeMap, message, out)
			} else {
				graph.processNode(ctx, rootNode, nodeMap, edgeMap, message, out)
			}
		}
	}()

	return out
}

func (graph *Graph) findRootNodes() []*Node {
	// Create a map of nodes with incoming edges
	hasIncoming := make(map[string]bool)
	for _, edge := range graph.Edges {
		if edge.Direction == DirectionTypeIn || edge.Direction == DirectionTypeBoth {
			hasIncoming[edge.To] = true
		}
	}

	// Return nodes with no incoming edges
	var rootNodes []*Node
	for _, node := range graph.Nodes {
		if !hasIncoming[node.ID] {
			rootNodes = append(rootNodes, node)
		}
	}
	return rootNodes
}

func (graph *Graph) processNode(
	ctx context.Context,
	node *Node,
	nodeMap map[string]*Node,
	edgeMap map[string][]*Edge,
	message *provider.Message,
	out chan<- provider.Event,
) {
	errnie.Log("Processing node: %s with message: %s", node.ID, message.Content)

	accumulator := provider.NewMessage(provider.RoleAssistant, "")

	for event := range node.Generate(ctx, message) {
		out <- event

		if event.Type == provider.EventChunk && event.Text != "" {
			accumulator.Append(event.Text)
		}

		if event.Type == provider.EventError {
			errnie.Log("Error in node %s: %v", node.ID, event.Error)
			return
		}
	}

	if accumulator.Content != "" {
		errnie.Log("Node %s accumulated response: %s", node.ID, accumulator.Content)
		nextMessage := provider.NewMessage(provider.RoleUser, accumulator.Content)

		for _, edge := range edgeMap[node.ID] {
			if nextNode, exists := nodeMap[edge.To]; exists {
				errnie.Log("Passing message from %s to %s", node.ID, edge.To)
				if nextNode.Parallel {
					go graph.processNode(ctx, nextNode, nodeMap, edgeMap, nextMessage, out)
				} else {
					graph.processNode(ctx, nextNode, nodeMap, edgeMap, nextMessage, out)
				}
			}
		}
	} else {
		errnie.Log("Node %s produced no content", node.ID)
	}
}
