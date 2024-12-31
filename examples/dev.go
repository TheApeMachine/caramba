package examples

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/system"
)

func RunDev() {
	ctx := context.Background()

	designAgent := ai.NewAgent(ctx, "architect", 1)
	implementAgent := ai.NewAgent(ctx, "developer", 2)
	testAgent := ai.NewAgent(ctx, "tester", 2)
	reviewAgent := ai.NewAgent(ctx, "reviewer", 1)

	designAgent.AddTools(provider.NewToolset("architect")...)
	implementAgent.AddTools(provider.NewToolset("developer")...)
	testAgent.AddTools(provider.NewToolset("tester")...)
	reviewAgent.AddTools(provider.NewToolset("reviewer")...)

	designAgent.Initialize()
	implementAgent.Initialize()
	testAgent.Initialize()
	reviewAgent.Initialize()

	// Create pipeline nodes
	node0 := system.NewNode("design", designAgent, false)
	node1 := system.NewNode("implement", implementAgent, true)
	node2 := system.NewNode("test", testAgent, true)
	node3 := system.NewNode("review", reviewAgent, false)

	edges := []*system.Edge{
		{From: "design", To: "implement", Direction: system.DirectionTypeBoth},
		{From: "implement", To: "test", Direction: system.DirectionTypeBoth},
		{From: "test", To: "review", Direction: system.DirectionTypeBoth},
	}

	graph := system.NewGraph([]*system.Node{node0, node1, node2, node3}, edges)

	fmt.Println("👩‍💻 Development Pipeline Example")
	fmt.Println("Enter the feature requirements:")

	var requirements string
	fmt.Scanln(&requirements)

	message := provider.NewMessage(provider.RoleUser, requirements)

	for event := range graph.Generate(ctx, message) {
		switch event.Type {
		case provider.EventChunk:
			if event.Text != "" {
				fmt.Print(event.Text)
			}
		case provider.EventToolCall:
			fmt.Printf("\n🛠  Using tool: %s\n", event.Name)
		case provider.EventError:
			fmt.Printf("\n❌ Error: %s\n", event.Error)
		}
	}
}
