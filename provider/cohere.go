package provider

import (
	"context"
	"io"

	cohereCore "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
)

type Cohere struct {
	client    *cohereclient.Client
	model     string
	maxTokens int
}

func NewCohere(apiKey string, model string) *Cohere {
	client := cohereclient.NewClient(
		cohereclient.WithToken(apiKey),
	)

	if apiKey == "" {
		return nil
	}
	if model == "" {
		model = "command" // Set default model if none specified
	}

	return &Cohere{
		client:    client,
		model:     model,
		maxTokens: 4096,
	}
}

func (c *Cohere) Generate(params GenerationParams) <-chan Event {
	out := make(chan Event)

	go func() {
		defer close(out)

		prompt := c.convertToCoherePrompt(params.Messages)

		stream, err := c.client.ChatStream(context.Background(), &cohereCore.ChatStreamRequest{
			Message: prompt,
			Model:   &c.model,
		})
		if err != nil {
			out <- Event{
				Type:  EventError,
				Error: err,
			}
			return
		}
		defer stream.Close()

		for {
			resp, err := stream.Recv()
			if err == io.EOF {
				out <- Event{Type: EventDone, Content: "\n"}
				break
			}
			if err != nil {
				out <- Event{
					Type:  EventError,
					Error: err,
				}
				break
			}

			if resp.TextGeneration != nil {
				out <- Event{
					Type:    EventToken,
					Content: resp.TextGeneration.Text,
				}
			}
		}
	}()

	return out
}

func (c *Cohere) convertToCoherePrompt(messages []Message) string {
	var prompt string
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			prompt += "System: " + msg.Content + "\n"
		case "user":
			prompt += "Human: " + msg.Content + "\n"
		case "assistant":
			prompt += "Assistant: " + msg.Content + "\n"
		}
	}
	return prompt
}
