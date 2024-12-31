package examples

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tools"
)

func RunChat() {
	ctx := context.Background()

	agent := ai.NewAgent(ctx, "assistant", 1)

	agent.AddTools(
		tools.NewBrowser(),
		tools.NewContainer(),
	)

	agent.Initialize()

	fmt.Println("💬 Simple Chat Example")
	fmt.Println("Type 'exit' to quit")
	fmt.Println("Enter your message:")

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}

		input := scanner.Text()
		if strings.ToLower(input) == "exit" {
			break
		}

		message := provider.NewMessage(provider.RoleUser, input)

		for event := range agent.Generate(ctx, message) {
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

	fmt.Println("\n👋 Goodbye!")
}
