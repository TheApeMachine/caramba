package examples

import (
	"bufio"
	"fmt"
	"os"
	"strings"

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

	rpcBaseURL := "localhost:3211" // JSON-RPC server (task submission)
	a2aBaseURL := "localhost:3210" // A2A HTTP server (SSE streaming)

	errnie.Info("Connecting to RPC server (task submission)...")
	rpcClient, err := client.NewRPCClient(
		client.WithBaseURL(rpcBaseURL),
	)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}
	errnie.Info("Connected to RPC server")

	streamClient, err := client.NewRPCClient(
		client.WithBaseURL(a2aBaseURL),
	)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	var (
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

		// Create a new request for each message
		taskObj := task.NewTask(
			task.WithMessages(
				task.NewSystemMessage("You are a helpful assistant"),
				task.NewUserMessage(example.name, input),
			),
		)
		req := task.NewTaskRequest(taskObj)

		// 1. Submit the task to the JSON-RPC server
		resp, err := rpcClient.SendTask(req, nil)
		if err != nil {
			errnie.New(errnie.WithError(fmt.Errorf("failed to submit task: %w", err)))
			continue
		}
		if resp == nil || resp.Result == nil {
			errnie.New(errnie.WithError(fmt.Errorf("no result from task submission")))
			continue
		}
		taskID := resp.Result.ID

		// 2. Stream the response from the A2A SSE endpoint
		streamReq := task.NewTaskRequest(&task.Task{ID: taskID})
		stream, err := streamClient.SendTaskStream(streamReq)
		if err != nil {
			errnie.New(errnie.WithError(fmt.Errorf("failed to connect to SSE endpoint: %w", err)))
			continue
		}
		for resp := range stream {
			if resp.Result != nil && resp.Result.Status.Message != nil {
				for _, part := range resp.Result.Status.Message.Parts {
					if textPart, ok := part.(*task.TextPart); ok {
						fmt.Print(textPart.Text)
					}
				}
			}
		}
		fmt.Println()
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
