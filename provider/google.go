package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/charmbracelet/log"
	"github.com/davecgh/go-spew/spew"
	"github.com/google/generative-ai-go/genai"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

type Gemini struct {
	*BaseProvider
	client *genai.Client
	model  *genai.GenerativeModel
	cancel context.CancelFunc
}

func NewGemini(apiKey string) *Gemini {
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		log.Error("Failed to create Gemini client", "error", err)
		return nil
	}
	model := client.GenerativeModel(viper.GetViper().GetString("models.gemini"))
	return &Gemini{
		BaseProvider: NewBaseProvider(),
		client:       client,
		model:        model,
	}
}

func (gemini *Gemini) Name() string {
	return "google (gemini)"
}

func (gemini *Gemini) Generate(params *LLMGenerationParams) <-chan *Event {
	errnie.Info("selected provider", "provider", "google", "model", gemini.model)

	out := make(chan *Event)
	ctx, cancel := context.WithCancel(context.Background())
	gemini.cancel = cancel

	chat := gemini.model.StartChat()

	go func() {
		defer close(out)
		defer cancel()

		startEvent := NewEvent("generate:start", EventStart, "gemini:gemini-1.5-flash", "", nil)
		out <- startEvent

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
		if len(params.Thread.Messages) > 0 {
			chat.History = make([]*genai.Content, 0, len(params.Thread.Messages))
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
					chat.History = append(chat.History, &genai.Content{
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
		if len(chat.History) == 0 {
			err := errors.New("no valid messages to process")
			log.Error("No valid messages to process", "error", err)
			errEvent := NewEvent("generate:error", EventError, err.Error(), "", nil)
			out <- errEvent
			return
		}

		// Get the last message for generation
		lastMessage := params.Thread.Messages[len(params.Thread.Messages)-1].Content
		stream := chat.SendMessageStream(ctx, genai.Text(lastMessage))

		for {
			select {
			case <-ctx.Done():
				return
			default:
				resp, err := stream.Next()
				if err == iterator.Done {
					doneEvent := NewEvent("generate:stop", EventStop, "", "", nil)
					out <- doneEvent
					return
				}
				if err != nil {
					log.Error("Error streaming Gemini response", "error", err)
					spew.Dump(params)
					errEvent := NewEvent("generate:error", EventError, err.Error(), "", nil)
					out <- errEvent
					return
				}

				if len(resp.Candidates) > 0 {
					for _, part := range resp.Candidates[0].Content.Parts {
						switch v := part.(type) {
						case genai.Text:
							chunkEvent := NewEvent("generate:contentblock:delta", EventChunk, string(v), "", nil)
							out <- chunkEvent
						case genai.FunctionCall:
							jsonArgs, err := json.Marshal(v.Args)
							if err != nil {
								errnie.Error(fmt.Errorf("failed to marshal function args: %w", err))
								errEvent := NewEvent("generate:error", EventError, fmt.Errorf("failed to marshal function args: %w", err).Error(), "", nil)
								out <- errEvent
								return
							}
							toolEvent := NewEvent("generate:toolcall", EventFunction, "Tool called", "", nil)
							toolEvent.PartialJSON = string(jsonArgs)
							out <- toolEvent
						}
					}
				}

				if resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != genai.BlockReasonUnspecified {
					errEvent := NewEvent("generate:error", EventError, fmt.Errorf("blocked: %v", resp.PromptFeedback.BlockReason).Error(), "", nil)
					out <- errEvent
					return
				}
			}
		}
	}()

	return out
}

func (gemini *Gemini) CancelGeneration(ctx context.Context) error {
	if gemini.cancel != nil {
		gemini.cancel()
	}
	return nil
}

// Version returns the provider version
func (g *Gemini) Version() string {
	return "1.0.0"
}

// Initialize sets up the provider
func (g *Gemini) Initialize(ctx context.Context) error {
	return nil
}

// PauseGeneration pauses the current generation
func (g *Gemini) PauseGeneration() error {
	return nil
}

// ResumeGeneration resumes the current generation
func (g *Gemini) ResumeGeneration() error {
	return nil
}

// GetCapabilities returns the provider capabilities
func (g *Gemini) GetCapabilities() map[string]interface{} {
	return map[string]interface{}{
		"streaming": true,
		"tools":     true,
	}
}

// SupportsFeature checks if a feature is supported
func (g *Gemini) SupportsFeature(feature string) bool {
	caps := g.GetCapabilities()
	supported, ok := caps[feature].(bool)
	return ok && supported
}

// ValidateConfig validates the provider configuration
func (g *Gemini) ValidateConfig() error {
	return nil
}

// Cleanup performs any necessary cleanup
func (g *Gemini) Cleanup(ctx context.Context) error {
	if g.client != nil {
		g.client.Close()
		g.client = nil
	}
	return nil
}

// Configure sets up the provider with the given configuration
func (g *Gemini) Configure(config *ProviderConfig) error {
	return nil
}

// GetConfig returns the current provider configuration
func (g *Gemini) GetConfig() *ProviderConfig {
	return nil
}

// GetMetrics returns the provider metrics
func (g *Gemini) GetMetrics() (*ProviderMetrics, error) {
	return &ProviderMetrics{}, nil
}

// HealthCheck performs a health check
func (g *Gemini) HealthCheck(ctx context.Context) *utils.HealthStatus {
	status := utils.StatusHealthy
	return &status
}
