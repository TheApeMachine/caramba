package devteam

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	devcfg "github.com/theapemachine/caramba/pkg/config"
)

/*
AnthropicProvider implements Provider using the Anthropic Messages API.
*/
type AnthropicProvider struct {
	client anthropic.Client
	model  string
}

/*
NewAnthropicProvider constructs an AnthropicProvider from a ProviderConfig.
*/
func NewAnthropicProvider(cfg devcfg.ProviderConfig) *AnthropicProvider {
	opts := []option.RequestOption{option.WithAPIKey(cfg.APIKey)}

	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}

	model := cfg.Model

	if model == "" {
		model = anthropic.ModelClaudeSonnet4_6
	}

	return &AnthropicProvider{
		client: anthropic.NewClient(opts...),
		model:  model,
	}
}

/*
Chat sends one conversation turn and returns the assistant reply.
*/
func (provider *AnthropicProvider) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	messages, err := provider.buildMessages(req.Messages)

	if err != nil {
		return ChatResponse{}, err
	}

	tools, err := provider.buildTools(req.Tools)

	if err != nil {
		return ChatResponse{}, err
	}

	maxTokens := int64(req.MaxTokens)

	if maxTokens == 0 {
		maxTokens = 8192
	}

	resp, err := provider.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     provider.model,
		MaxTokens: maxTokens,
		System:    []anthropic.TextBlockParam{{Text: req.System}},
		Tools:     tools,
		Messages:  messages,
	})

	if err != nil {
		return ChatResponse{}, fmt.Errorf("anthropic: %w", err)
	}

	return provider.parseResponse(resp)
}

func (provider *AnthropicProvider) buildMessages(
	msgs []ChatMessage,
) ([]anthropic.MessageParam, error) {
	params := make([]anthropic.MessageParam, 0, len(msgs))

	for _, msg := range msgs {
		switch msg.Role {
		case "user":
			params = append(params, anthropic.NewUserMessage(anthropic.NewTextBlock(msg.Content)))

		case "assistant":
			blocks := make([]anthropic.ContentBlockParamUnion, 0)

			if msg.Content != "" {
				blocks = append(blocks, anthropic.NewTextBlock(msg.Content))
			}

			for _, tc := range msg.ToolCalls {
				blocks = append(blocks, anthropic.NewToolUseBlock(tc.ID, tc.Input, tc.Name))
			}

			params = append(params, anthropic.MessageParam{
				Role:    anthropic.MessageParamRoleAssistant,
				Content: blocks,
			})

		case "tool":
			params = append(params, anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(msg.ToolCallID, msg.Content, false),
			))

		default:
			return nil, fmt.Errorf("anthropic: unknown message role %q", msg.Role)
		}
	}

	return params, nil
}

func (provider *AnthropicProvider) buildTools(
	defs []ToolDefinition,
) ([]anthropic.ToolUnionParam, error) {
	tools := make([]anthropic.ToolUnionParam, len(defs))

	for index, def := range defs {
		raw, err := json.Marshal(def.Parameters)

		if err != nil {
			return nil, fmt.Errorf("anthropic: marshal tool %q parameters: %w", def.Name, err)
		}

		var schema anthropic.ToolInputSchemaParam

		if err := json.Unmarshal(raw, &schema); err != nil {
			return nil, fmt.Errorf("anthropic: unmarshal tool %q schema: %w", def.Name, err)
		}

		tp := anthropic.ToolParam{
			Name:        def.Name,
			Description: anthropic.String(def.Description),
			InputSchema: schema,
		}

		tools[index] = anthropic.ToolUnionParam{OfTool: &tp}
	}

	return tools, nil
}

func (provider *AnthropicProvider) parseResponse(resp *anthropic.Message) (ChatResponse, error) {
	result := ChatResponse{}

	for _, block := range resp.Content {
		toolUse := block.AsToolUse()

		if toolUse.ID != "" {
			var input map[string]any

			if err := json.Unmarshal(toolUse.Input, &input); err != nil {
				return ChatResponse{}, fmt.Errorf("anthropic: unmarshal tool call %q input: %w", toolUse.Name, err)
			}

			result.ToolCalls = append(result.ToolCalls, ToolCall{
				ID:    toolUse.ID,
				Name:  toolUse.Name,
				Input: input,
			})

			continue
		}

		result.Content += block.AsText().Text
	}

	return result, nil
}
