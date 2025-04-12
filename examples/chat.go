package examples

import (
	"fmt"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/service/client"
	"github.com/theapemachine/caramba/pkg/task"
)

type ChatExample struct {
	name string
}

func NewChatExample(name string) *ChatExample {
	return &ChatExample{
		name: name,
	}
}

func (example *ChatExample) Run() error {
	errnie.Info("Terminal Chat Agent with A2A/MCP Integration")
	errnie.Info("Type 'exit' to quit or 'help' for commands")

	client := client.NewA2AClient("http://localhost:3210")

	for {
		var input string

		fmt.Print("$caramba> ")
		fmt.Scanln(&input)

		if input == "exit" {
			break
		}

		client.SendTask(
			uuid.New().String(),
			task.NewUserMessage(example.name, input),
		)
	}

	return nil
}
