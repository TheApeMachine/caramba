package examples

import (
	"fmt"
	"os"

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

	client := client.NewA2AClient(
		client.WithBaseURL("http://localhost:3210"),
	)

	var (
		request  = task.NewTaskRequest(task.NewTask())
		response = new(task.TaskResponse)
		err      error
	)

	for {
		var input string

		fmt.Print("$" + example.name + "> ")
		fmt.Scanln(&input)

		if input == "exit" {
			break
		}

		request.AddMessage(
			task.NewUserMessage(example.name, input),
		)

		// Create and send task
		response, err = client.SendTask(
			request, os.Stdout,
		)

		if err != nil {
			errnie.Error("Error sending task", errnie.WithError(err))
		}

		if response != nil {
			request.AddResult(response)
		}

		for _, message := range response.Result.History {
			for _, part := range message.Parts {
				fmt.Print(part.Text)
			}
		}
	}

	return nil
}
