package examples

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/system"
	"github.com/theapemachine/caramba/tools"
)

func RunResearch() {
	ctx := context.Background()

	inputAgent := ai.NewAgent(ctx, "prompt", 1)
	searchAgent := ai.NewAgent(ctx, "researcher", 2)
	analyzeAgent := ai.NewAgent(ctx, "analyst", 2)
	writeAgent := ai.NewAgent(ctx, "writer", 1)

	searchAgent.AddTools(
		tools.NewBrowser(),
		tools.NewQdrantStore("research", 1536),
	)
	analyzeAgent.AddTools(
		tools.NewQdrantQuery("research", 1536),
	)

	inputAgent.Initialize()
	searchAgent.Initialize()
	analyzeAgent.Initialize()
	writeAgent.Initialize()

	node0 := system.NewNode("input", inputAgent, false)
	node1 := system.NewNode("search", searchAgent, true)
	node2 := system.NewNode("analyze", analyzeAgent, false)
	node3 := system.NewNode("write", writeAgent, false)

	edges := []*system.Edge{
		{From: "input", To: "search", Direction: system.DirectionTypeOut},
		{From: "search", To: "analyze", Direction: system.DirectionTypeOut},
		{From: "analyze", To: "write", Direction: system.DirectionTypeOut},
	}

	graph := system.NewGraph([]*system.Node{node0, node1, node2, node3}, edges)

	fmt.Println("🔍 Research Pipeline Example")
	fmt.Println("Enter your research question:")

	var question string
	fmt.Scanln(&question)

	message := provider.NewMessage(provider.RoleUser, question)

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
