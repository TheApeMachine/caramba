package examples

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/service/client"
	"github.com/theapemachine/caramba/pkg/stores/inmemory"
	"github.com/theapemachine/caramba/pkg/stores/types"
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
	errnie.Info("Connecting to RPC server...")

	client, err := client.NewRPCClient(
		client.WithBaseURL("localhost:3211"),
	)

	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	errnie.Info("Connected to RPC server")

	var (
		request = task.NewTaskRequest(task.NewTask(
			task.WithMessages(
				task.NewSystemMessage("You are a helpful assistant"),
			),
		))

		reader = bufio.NewReader(os.Stdin)
	)

	for {
		fmt.Print("$" + example.name + "> ")
		input, err := reader.ReadString('\n')

		if err != nil {
			return errnie.New(errnie.WithError(err))
		}

		input = strings.TrimSpace(input)

		if input == "" {
			continue
		}

		if input == "exit" {
			break
		}

		if input == "help" {
			example.printHelp()
			continue
		}

		request.Params.AddMessage(task.NewUserMessage(example.name, input))

		response, err := client.SendTask(request, os.Stdout)

		if err != nil {
			errnie.New(errnie.WithError(err))
			continue
		}

		session := inmemory.NewSession(types.NewQuery(
			types.WithFilter("id", response.Result.SessionID),
		))

		outTask := task.NewTask()

		for outTask.Status.State != task.TaskStateCompleted {
			if _, err := io.Copy(outTask, session); err != nil {
				return errnie.New(errnie.WithError(err))
			}

			time.Sleep(100 * time.Millisecond)
		}
	}

	return nil
}

func (example *ChatExample) printHelp() {
	fmt.Println("Available commands:")
	fmt.Println("  help  - Show this help message")
	fmt.Println("  exit  - Exit the chat")
	fmt.Println("")
	fmt.Println("Just type your message and press Enter to chat with the assistant.")
}
