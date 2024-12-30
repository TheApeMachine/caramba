package provider

import (
	"context"
	"errors"
	"net/http"
	"net/url"

	sdk "github.com/ollama/ollama/api"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

type Ollama struct {
	client *sdk.Client
	model  string
}

func NewOllama(host string) *Ollama {
	hostURL, _ := url.Parse(host)
	return &Ollama{
		client: sdk.NewClient(hostURL, http.DefaultClient),
		model:  "llama3.2:3b",
	}
}

func (ollama *Ollama) Name() string {
	return "ollama (llama3.2:3b)"
}

func (ollama *Ollama) Generate(ctx context.Context, params *GenerationParams) <-chan Event {
	out := make(chan Event)

	go func() {
		defer close(out)

		// Convert our tools to Ollama format only if tools exist
		var tools []sdk.Tool
		if len(params.Tools) > 0 {
			tools = make([]sdk.Tool, len(params.Tools))
			for i, tool := range params.Tools {
				tools[i] = sdk.Tool{
					Type: "function",
					Function: sdk.ToolFunction{
						Name:        tool.Name,
						Description: tool.Description,
						Parameters:  tool.Schema.(sdk.ToolFunction).Parameters,
					},
				}
			}
		}

		// Convert our messages to Ollama format only if messages exist
		var messages []sdk.Message
		if len(params.Thread.Messages) > 0 {
			messages = make([]sdk.Message, len(params.Thread.Messages))
			for i, msg := range params.Thread.Messages {
				if msg.Content != "" {
					messages[i] = sdk.Message{
						Role:    string(msg.Role),
						Content: msg.Content,
					}
				}
			}
		}

		// Only proceed if we have messages
		if len(messages) == 0 {
			errnie.Error(errors.New("no valid messages to process"))
			out <- Event{Type: EventError, Error: errors.New("no valid messages to process")}
			return
		}

		// Build request with only non-empty fields
		request := &sdk.ChatRequest{
			Model:    ollama.model,
			Messages: messages,
			Stream:   utils.BoolPtr(true),
		}

		// Only add tools if we have any
		if len(tools) > 0 {
			request.Tools = sdk.Tools(tools)
		}

		err := ollama.client.Chat(ctx, request, func(resp sdk.ChatResponse) error {
			if resp.Message.Content != "" {
				out <- Event{Type: EventChunk, Text: resp.Message.Content}
			}
			return nil
		})

		if err != nil {
			errnie.Error(err)
			out <- Event{Type: EventError, Error: err}
			return
		}

		out <- Event{Type: EventDone, Text: "\n"}
	}()

	return out
}
