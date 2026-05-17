package devteam

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	openai "github.com/openai/openai-go"
	openaiopt "github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"

	devcfg "github.com/theapemachine/caramba/pkg/config"
)

/*
OpenAIProvider implements Provider using OpenAI Responses for the default
OpenAI endpoint and Chat Completions for OpenAI-compatible endpoints.
*/
type OpenAIProvider struct {
	client             openai.Client
	model              string
	useChatCompletions bool
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
		client:             openai.NewClient(opts...),
		model:              model,
		useChatCompletions: cfg.BaseURL != "" || (cfg.Provider != "" && strings.ToLower(cfg.Provider) != "openai"),
	}
}

/*
Chat sends one conversation turn via the Responses API and returns the reply.
*/
func (provider *OpenAIProvider) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	if provider.useChatCompletions {
		return runProviderChat(ctx, func(requestCtx context.Context) (ChatResponse, error) {
			return provider.chatCompletions(requestCtx, req)
		})
	}

	return runProviderChat(ctx, func(requestCtx context.Context) (ChatResponse, error) {
		return provider.responses(requestCtx, req)
	})
}

func (provider *OpenAIProvider) responses(
	ctx context.Context,
	req ChatRequest,
) (ChatResponse, error) {
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

func (provider *OpenAIProvider) chatCompletions(
	ctx context.Context,
	req ChatRequest,
) (ChatResponse, error) {
	messages, err := provider.buildChatMessages(req)

	if err != nil {
		return ChatResponse{}, err
	}

	tools, err := provider.buildChatTools(req.Tools)

	if err != nil {
		return ChatResponse{}, err
	}

	maxTokens := int64(req.MaxTokens)

	if maxTokens == 0 {
		maxTokens = 8192
	}

	params := openai.ChatCompletionNewParams{
		Model:     openai.ChatModel(provider.model),
		Messages:  messages,
		MaxTokens: openai.Int(maxTokens),
	}

	if len(tools) > 0 {
		params.Tools = tools
	}

	resp, err := provider.client.Chat.Completions.New(ctx, params)

	if err != nil {
		return ChatResponse{}, fmt.Errorf("openai: %w", err)
	}

	return provider.parseChatCompletion(resp)
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

func (provider *OpenAIProvider) buildChatMessages(
	req ChatRequest,
) ([]openai.ChatCompletionMessageParamUnion, error) {
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(req.Messages)+1)

	if strings.TrimSpace(req.System) != "" {
		messages = append(messages, openai.SystemMessage(req.System))
	}

	for _, msg := range req.Messages {
		switch msg.Role {
		case "user":
			messages = append(messages, openai.UserMessage(msg.Content))

		case "assistant":
			message, err := provider.chatAssistantMessage(msg)

			if err != nil {
				return nil, err
			}

			messages = append(messages, message)

		case "tool":
			messages = append(messages, openai.ToolMessage(msg.Content, msg.ToolCallID))

		default:
			return nil, fmt.Errorf("openai: unknown message role %q", msg.Role)
		}
	}

	return messages, nil
}

func (provider *OpenAIProvider) chatAssistantMessage(
	msg ChatMessage,
) (openai.ChatCompletionMessageParamUnion, error) {
	if len(msg.ToolCalls) == 0 {
		return openai.AssistantMessage(msg.Content), nil
	}

	assistant := openai.ChatCompletionAssistantMessageParam{}

	if msg.Content != "" {
		assistant.Content.OfString = openai.String(msg.Content)
	}

	assistant.ToolCalls = make([]openai.ChatCompletionMessageToolCallParam, 0, len(msg.ToolCalls))

	for _, toolCall := range msg.ToolCalls {
		args, err := json.Marshal(toolCall.Input)

		if err != nil {
			return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf(
				"openai: marshal tool call %q arguments: %w",
				toolCall.Name,
				err,
			)
		}

		assistant.ToolCalls = append(
			assistant.ToolCalls,
			openai.ChatCompletionMessageToolCallParam{
				ID: toolCall.ID,
				Function: openai.ChatCompletionMessageToolCallFunctionParam{
					Arguments: string(args),
					Name:      toolCall.Name,
				},
			},
		)
	}

	return openai.ChatCompletionMessageParamUnion{OfAssistant: &assistant}, nil
}

func (provider *OpenAIProvider) buildChatTools(
	defs []ToolDefinition,
) ([]openai.ChatCompletionToolParam, error) {
	tools := make([]openai.ChatCompletionToolParam, len(defs))

	for index, def := range defs {
		raw, err := json.Marshal(def.Parameters)

		if err != nil {
			return nil, fmt.Errorf("openai: marshal tool %q parameters: %w", def.Name, err)
		}

		var params shared.FunctionParameters

		if err := json.Unmarshal(raw, &params); err != nil {
			return nil, fmt.Errorf("openai: unmarshal tool %q parameters: %w", def.Name, err)
		}

		tools[index] = openai.ChatCompletionToolParam{
			Function: shared.FunctionDefinitionParam{
				Name:        def.Name,
				Description: openai.String(def.Description),
				Parameters:  params,
			},
		}
	}

	return tools, nil
}

func (provider *OpenAIProvider) parseResponse(resp *responses.Response) (ChatResponse, error) {
	result := ChatResponse{
		Content:      resp.OutputText(),
		InputTokens:  resp.Usage.InputTokens,
		OutputTokens: resp.Usage.OutputTokens,
		TotalTokens:  resp.Usage.TotalTokens,
	}

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

func (provider *OpenAIProvider) parseChatCompletion(
	resp *openai.ChatCompletion,
) (ChatResponse, error) {
	result := ChatResponse{
		InputTokens:  resp.Usage.PromptTokens,
		OutputTokens: resp.Usage.CompletionTokens,
		TotalTokens:  resp.Usage.TotalTokens,
	}

	if len(resp.Choices) == 0 {
		return result, nil
	}

	message := resp.Choices[0].Message
	result.Content = message.Content

	for _, toolCall := range message.ToolCalls {
		var input map[string]any

		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &input); err != nil {
			return ChatResponse{}, fmt.Errorf(
				"openai: unmarshal tool call %q arguments: %w",
				toolCall.Function.Name,
				err,
			)
		}

		result.ToolCalls = append(result.ToolCalls, ToolCall{
			ID:    toolCall.ID,
			Name:  toolCall.Function.Name,
			Input: input,
		})
	}

	return result, nil
}
