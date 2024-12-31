package examples

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/system"
)

func RunStrawberry() {
	// Create a simple test graph with 3 nodes
	graph := createTestGraph()

	// Create a properly formatted test message
	message := provider.NewMessage(provider.RoleUser, `How many times do we find the letter r in the word strawberry?`)

	// Create context
	ctx := context.Background()

	// Create a consumer for pretty printing
	consumer := stream.NewConsumer()

	// Print formatted responses
	fmt.Println("\n=== Agent Responses ===")
	consumer.Print(graph.Generate(ctx, message), false)

}

/*
createTestGraph is a function that creates a simple test graph with 3 nodes.
*/
func createTestGraph() *system.Graph {
	ctx := context.Background()

	node0 := system.NewNode("node0", ai.NewAgent(ctx, "prompt", 1), false)
	node1 := system.NewNode("node1", ai.NewAgent(ctx, "reasoner", 2), false)
	node2 := system.NewNode("node2", ai.NewAgent(ctx, "challenger", 2), false)
	node3 := system.NewNode("node3", ai.NewAgent(ctx, "solver", 2), false)

	node0.Agent.Initialize()
	node1.Agent.Initialize()
	node2.Agent.Initialize()
	node3.Agent.Initialize()

	// node0.Agent.AddProcess(provider.NewCompoundProcess([]provider.Process{
	// 	&prompt.Process{},
	// }))
	// node1.Agent.AddProcess(provider.NewCompoundProcess([]provider.Process{
	// 	&reasoning.Process{},
	// 	&reflection.Process{},
	// }))
	// node2.Agent.AddProcess(provider.NewCompoundProcess([]provider.Process{
	// 	&reasoning.Process{},
	// 	&reflection.Process{},
	// }))
	// node3.Agent.AddProcess(provider.NewCompoundProcess([]provider.Process{
	// 	&reasoning.Process{},
	// 	&reflection.Process{},
	// }))

	edge0 := &system.Edge{
		From:      "node0",
		To:        "node1",
		Direction: system.DirectionTypeOut,
	}
	edge1 := &system.Edge{
		From:      "node1",
		To:        "node2",
		Direction: system.DirectionTypeOut,
	}
	edge2 := &system.Edge{
		From:      "node2",
		To:        "node3",
		Direction: system.DirectionTypeOut,
	}

	return &system.Graph{
		Nodes: []*system.Node{node0, node1, node2, node3},
		Edges: []*system.Edge{edge0, edge1, edge2},
	}
}
