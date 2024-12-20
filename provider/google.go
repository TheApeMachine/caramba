package provider

import (
	"context"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

type Google struct {
	client    *genai.Client
	model     string
	maxTokens int
}

func NewGoogle(apiKey string, model string) *Google {
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil
	}

	return &Google{
		client:    client,
		model:     model,
		maxTokens: 4096,
	}
}

func (g *Google) Generate(params GenerationParams) <-chan Event {
	out := make(chan Event)

	go func() {
		defer close(out)

		model := g.client.GenerativeModel(g.model)
		temp := float32(params.Temperature)
		model.Temperature = &temp

		parts := g.convertToGoogleParts(params.Messages)
		iter := model.GenerateContentStream(context.Background(), parts...)

		for {
			resp, err := iter.Next()
			if err == iterator.Done {
				out <- Event{Type: EventDone}
				break
			}

			if err != nil {
				out <- Event{
					Type:  EventError,
					Error: err,
				}
				break
			}

			for _, part := range resp.Candidates[0].Content.Parts {
				if text, ok := part.(genai.Text); ok {
					out <- Event{
						Type:    EventToken,
						Content: string(text),
					}
				}
			}
		}
	}()

	return out
}

func (g *Google) convertToGoogleParts(messages []Message) []genai.Part {
	var parts []genai.Part

	for _, msg := range messages {
		content := genai.Content{
			Parts: []genai.Part{genai.Text(msg.Content)},
		}
		parts = append(parts, content.Parts...)
	}

	return parts
}
