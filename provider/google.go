package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/google/generative-ai-go/genai"
	"github.com/theapemachine/errnie"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

type Gemini struct {
	client *genai.Client
	model  *genai.GenerativeModel
}

func NewGemini(apiKey string) *Gemini {
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		errnie.Error(fmt.Errorf("failed to create genai client: %w", err))
		return nil
	}
	model := client.GenerativeModel("gemini-pro")
	return &Gemini{
		client: client,
		model:  model,
	}
}

func (gemini *Gemini) Name() string {
	return "google (gemini)"
}

func (gemini *Gemini) Generate(ctx context.Context, params *GenerationParams) <-chan Event {
	out := make(chan Event)

	go func() {
		defer close(out)

		// Convert our tools to Gemini tool format only if tools exist
		var tools []*genai.Tool
		if len(params.Tools) > 0 {
			tools = make([]*genai.Tool, len(params.Tools))
			for i, tool := range params.Tools {
				// Fix parameters type conversion
				parameters := make(map[string]*genai.Schema)
				if toolSchema, ok := tool.GenerateSchema().(map[string]interface{}); ok {
					for k, v := range toolSchema {
						if schemaMap, ok := v.(map[string]interface{}); ok {
							var schemaType genai.Type
							switch schemaMap["type"].(string) {
							case "string":
								schemaType = genai.TypeString
							case "number":
								schemaType = genai.TypeNumber
							case "integer":
								schemaType = genai.TypeInteger
							case "boolean":
								schemaType = genai.TypeBoolean
							case "array":
								schemaType = genai.TypeArray
							case "object":
								schemaType = genai.TypeObject
							default:
								schemaType = genai.TypeUnspecified
							}

							parameters[k] = &genai.Schema{
								Type:        schemaType,
								Description: schemaMap["description"].(string),
							}
						}
					}
				}

				tools[i] = &genai.Tool{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{
							Name:        tool.Name(),
							Description: tool.Description(),
							Parameters: &genai.Schema{
								Type:       genai.TypeObject,
								Properties: parameters,
							},
						},
					},
				}
			}
		}

		// Convert our messages to Gemini content format
		var geminiMessages []*genai.Content
		if len(params.Thread.Messages) > 0 {
			geminiMessages = make([]*genai.Content, 0, len(params.Thread.Messages))
			for _, msg := range params.Thread.Messages {
				var role string
				switch msg.Role {
				case RoleSystem:
					role = "user"
				case RoleUser:
					role = "user"
				case RoleAssistant:
					role = "model"
				default:
					continue // Skip invalid roles
				}

				if msg.Content != "" {
					geminiMessages = append(geminiMessages, &genai.Content{
						Parts: []genai.Part{
							genai.Text(msg.Content),
						},
						Role: role,
					})
				}
			}
		}

		// Add tool config to model if tools exist
		if len(tools) > 0 {
			gemini.model.Tools = tools
		}

		// Only proceed if we have messages
		if len(geminiMessages) == 0 {
			errnie.Error(errors.New("no valid messages to process"))
			out <- Event{Type: EventError, Error: errors.New("no valid messages to process")}
			return
		}

		// Get the last message for generation
		lastMessage := params.Thread.Messages[len(params.Thread.Messages)-1].Content
		stream := gemini.model.GenerateContentStream(ctx, genai.Text(lastMessage))

		for {
			resp, err := stream.Next()
			if err == iterator.Done {
				out <- Event{Type: EventDone, Text: "\n"}
				return
			}
			if err != nil {
				errnie.Error(err)
				out <- Event{Type: EventError, Error: err}
				return
			}

			if len(resp.Candidates) > 0 {
				for _, part := range resp.Candidates[0].Content.Parts {
					switch v := part.(type) {
					case genai.Text:
						out <- Event{Type: EventChunk, Text: string(v)}
					case genai.FunctionCall:
						jsonArgs, err := json.Marshal(v.Args)
						if err != nil {
							errnie.Error(fmt.Errorf("failed to marshal function args: %w", err))
							out <- Event{Type: EventError, Error: fmt.Errorf("failed to marshal function args: %w", err)}
							return
						}
						out <- Event{Type: EventChunk, PartialJSON: string(jsonArgs)}
					}
				}
			}

			if resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != genai.BlockReasonUnspecified {
				out <- Event{Type: EventError, Error: fmt.Errorf("blocked: %v", resp.PromptFeedback.BlockReason)}
				return
			}
		}
	}()

	return out
}
