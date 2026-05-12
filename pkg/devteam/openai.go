package devteam

import (
	"context"
	"encoding/json"
	"fmt"

	openai "github.com/openai/openai-go"
	openaiopt "github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"

	devcfg "github.com/theapemachine/caramba/pkg/config"
)

/*
OpenAIProvider implements Provider using the OpenAI Responses API.
Because the SDK accepts a custom BaseURL it works with any OpenAI-compatible
endpoint: OpenAI, Groq, Together, Ollama, vLLM, LM Studio, etc.
*/
type OpenAIProvider struct {
	client openai.Client
	model  string
}

/*
NewOpenAIProvider constructs an OpenAIProvider from a ProviderConfig.
*/
func NewOpenAIProvider(cfg devcfg.ProviderConfig) *OpenAIProvider {
	opts := []openaiopt.RequestOption{openaiopt.WithAPIKey(cfg.APIKey)}

	if cfg.BaseURL != "" {
		opts = append(opts, openaiopt.WithBaseURL(cfg.BaseURL))
	}

	model := cfg.Model

	if model == "" {
		model = "gpt-4o"
	}

	return &OpenAIProvider{
		client: openai.NewClient(opts...),
		model:  model,
	}
}

/*
Chat sends one conversation turn via the Responses API and returns the reply.
*/
func (provider *OpenAIProvider) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	input, err := provider.buildInput(req.Messages)

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

	params := responses.ResponseNewParams{
		Model:           provider.model,
		Input:           responses.ResponseNewParamsInputUnion{OfInputItemList: input},
		MaxOutputTokens: openai.Int(maxTokens),
	}

	if req.System != "" {
		params.Instructions = openai.String(req.System)
	}

	if len(tools) > 0 {
		params.Tools = tools
	}

	resp, err := provider.client.Responses.New(ctx, params)

	if err != nil {
		return ChatResponse{}, fmt.Errorf("openai: %w", err)
	}

	return provider.parseResponse(resp)
}

func (provider *OpenAIProvider) buildInput(
	msgs []ChatMessage,
) (responses.ResponseInputParam, error) {
	items := make(responses.ResponseInputParam, 0, len(msgs))

	for _, msg := range msgs {
		switch msg.Role {
		case "user":
			items = append(items, responses.ResponseInputItemParamOfMessage(
				msg.Content, responses.EasyInputMessageRoleUser,
			))

		case "assistant":
			if len(msg.ToolCalls) == 0 {
				items = append(items, responses.ResponseInputItemParamOfMessage(
					msg.Content, responses.EasyInputMessageRoleAssistant,
				))

				continue
			}

			for _, tc := range msg.ToolCalls {
				args, err := json.Marshal(tc.Input)

				if err != nil {
					return nil, fmt.Errorf("openai: marshal tool call %q arguments: %w", tc.Name, err)
				}

				items = append(items, responses.ResponseInputItemParamOfFunctionCall(
					string(args), tc.ID, tc.Name,
				))
			}

		case "tool":
			items = append(items, responses.ResponseInputItemParamOfFunctionCallOutput(
				msg.ToolCallID, msg.Content,
			))

		default:
			return nil, fmt.Errorf("openai: unknown message role %q", msg.Role)
		}
	}

	return items, nil
}

func (provider *OpenAIProvider) buildTools(
	defs []ToolDefinition,
) ([]responses.ToolUnionParam, error) {
	tools := make([]responses.ToolUnionParam, len(defs))

	for index, def := range defs {
		raw, err := json.Marshal(def.Parameters)

		if err != nil {
			return nil, fmt.Errorf("openai: marshal tool %q parameters: %w", def.Name, err)
		}

		var params openai.FunctionParameters

		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, fmt.Errorf("openai: unmarshal tool %q parameters: %w", def.Name, err)
		}

		tools[index] = responses.ToolParamOfFunction(def.Name, params, false)
	}

	return tools, nil
}

func (provider *OpenAIProvider) parseResponse(resp *responses.Response) (ChatResponse, error) {
	result := ChatResponse{Content: resp.OutputText()}

	for _, item := range resp.Output {
		fc := item.AsFunctionCall()

		if fc.CallID == "" {
			continue
		}

		var input map[string]any

		if err := json.Unmarshal([]byte(fc.Arguments), &input); err != nil {
			return ChatResponse{}, fmt.Errorf("openai: unmarshal tool call %q arguments: %w", fc.Name, err)
		}

		result.ToolCalls = append(result.ToolCalls, ToolCall{
			ID:    fc.CallID,
			Name:  fc.Name,
			Input: input,
		})
	}

	return result, nil
}
