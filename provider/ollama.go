package provider

import (
	"context"
	"net/http"
	"net/url"

	"github.com/ollama/ollama/api"
)

func Bool(b bool) *bool {
	return &b
}

type Ollama struct {
	client *api.Client
	model  string
	system string
}

func NewOllama(model string) *Ollama {
	client := api.NewClient(
		&url.URL{Scheme: "http", Host: "localhost:11434"},
		&http.Client{},
	)

	return &Ollama{
		client: client,
		model:  model,
	}
}

func (o *Ollama) Generate(params GenerationParams) <-chan Event {
	out := make(chan Event)

	go func() {
		defer close(out)

		prompt := o.convertToOllamaPrompt(params.Messages)

		req := &api.GenerateRequest{
			Model:  o.model,
			Prompt: prompt,
			Stream: Bool(true),
			Options: map[string]interface{}{
				"temperature": params.Temperature,
			},
		}

		respFunc := func(resp api.GenerateResponse) error {
			out <- Event{
				Type:    EventToken,
				Content: resp.Response,
			}
			return nil
		}

		if err := o.client.Generate(context.Background(), req, respFunc); err != nil {
			out <- Event{
				Type:  EventError,
				Error: err,
			}
			return
		}

		out <- Event{Type: EventDone}
	}()

	return out
}

func (o *Ollama) convertToOllamaPrompt(messages []Message) string {
	var prompt string

	// Add system message if available
	if o.system != "" {
		prompt += "System: " + o.system + "\n"
	}

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			o.system = msg.Content
		case "user":
			prompt += "Human: " + msg.Content + "\n"
		case "assistant":
			prompt += "Assistant: " + msg.Content + "\n"
		}
	}

	return prompt
}

func (o *Ollama) Configure(config map[string]interface{}) {
	if systemMsg, ok := config["system_message"].(string); ok {
		o.system = systemMsg
	}
	if model, ok := config["model"].(string); ok {
		o.model = model
	}
}
