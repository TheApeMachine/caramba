package provider

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/davecgh/go-spew/spew"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ProviderService struct {
	client *openai.Client
	params *ProviderParams
	ctx    context.Context
}

func NewProvider() *ProviderService {
	client := openai.NewClient(option.WithAPIKey(os.Getenv("OPENAI_API_KEY")))

	return &ProviderService{
		client: &client,
		params: &ProviderParams{},
		ctx:    context.Background(),
	}
}

func (prvdr *ProviderService) SetParams(params *ProviderParams) {
	prvdr.params = params
}

func (prvdr *ProviderService) Generate() error {
	errnie.Debug("provider.Generate")

	model, err := prvdr.params.Model()

	if err != nil {
		return errnie.Error(err)
	}

	composed := &openai.ChatCompletionNewParams{
		Model:            model,
		Temperature:      openai.Float(prvdr.params.Temperature()),
		TopP:             openai.Float(prvdr.params.TopP()),
		FrequencyPenalty: openai.Float(prvdr.params.FrequencyPenalty()),
		PresencePenalty:  openai.Float(prvdr.params.PresencePenalty()),
	}

	if maxTokens := prvdr.params.MaxTokens(); maxTokens > 1 {
		composed.MaxTokens = openai.Int(int64(maxTokens))
	}

	// Build messages and tools using helper functions.
	for _, buildFunc := range []func(*openai.ChatCompletionNewParams) error{
		prvdr.buildMessages,
		prvdr.buildTools,
	} {
		if err := buildFunc(composed); err != nil {
			return errnie.Error(err)
		}
	}

	if prvdr.params.HasResponseFormat() {
		if err := prvdr.buildResponseFormat(composed); err != nil {
			return errnie.Error(err)
		}
	}

	return prvdr.handleSingleRequest(composed)
}

func (prvdr *ProviderService) handleSingleRequest(params *openai.ChatCompletionNewParams) error {
	errnie.Debug("provider.handleSingleRequest")

	// Ensure all assistant messages with tool calls have corresponding tool responses
	// before sending to the OpenAI API
	cleanedMessages := prvdr.ensureToolResponses(params.Messages)
	params.Messages = cleanedMessages

	for _, msg := range params.Messages {
		errnie.Debug("message", "message", msg)
	}

	completion, err := prvdr.client.Chat.Completions.New(context.Background(), *params)

	if err != nil {
		spew.Dump(params)
		return errnie.Error(err)
	}

	messages, err := prvdr.params.Messages()

	if err != nil {
		return errnie.Error(err)
	}

	msgList, err := NewMessage_List(prvdr.params.Segment(), int32(messages.Len()+1))

	if err != nil {
		return errnie.Error(err)
	}

	// Copy over existing messages.
	for i := range messages.Len() {
		msg := messages.At(i)
		msgList.Set(i, msg)
	}

	toolCalls := completion.Choices[0].Message.ToolCalls

	if len(toolCalls) > 0 {
		// Append the assistant message with tool calls.
		params.Messages = append(params.Messages, completion.Choices[0].Message.ToParam())

		toolCallsList, err := NewToolCall_List(prvdr.params.Segment(), int32(len(toolCalls)))

		if err != nil {
			return errnie.Error(err)
		}

		for i, tc := range toolCalls {
			errnie.Debug("toolCall", "tool", tc.Function.Name, "id", tc.ID)
			tcItem := toolCallsList.At(i)

			if err := tcItem.SetId(tc.ID); err != nil {
				return errnie.Error(err)
			}

			if err := tcItem.SetType("function"); err != nil {
				return errnie.Error(err)
			}

			tcFunc, err := tcItem.NewFunction()

			if err != nil {
				return errnie.Error(err)
			}

			if err := tcFunc.SetName(tc.Function.Name); err != nil {
				return errnie.Error(err)
			}

			if err := tcFunc.SetArguments(tc.Function.Arguments); err != nil {
				return errnie.Error(err)
			}

			errnie.Info("provider.handleSingleRequest", "toolcall", tc.Function.Name, "arguments", tc.Function.Arguments)
		}

		msg, err := NewMessage(prvdr.params.Segment())

		if err != nil {
			return errnie.Error(err)
		}

		msg.SetRole("assistant")
		msg.SetToolCalls(toolCallsList)
		msg.SetContent(completion.Choices[0].Message.Content)
		msgList.Set(messages.Len(), msg)

		return prvdr.params.SetMessages(msgList)
	}

	// Normal case without tool calls.
	msg, err := NewMessage(prvdr.params.Segment())
	if err != nil {
		return errnie.Error(err)
	}
	msg.SetRole("assistant")
	msg.SetContent(completion.Choices[0].Message.Content)
	msgList.Set(messages.Len(), msg)

	return prvdr.params.SetMessages(msgList)
}

func (prvdr *ProviderService) handleStreamingRequest(params *openai.ChatCompletionNewParams) error {
	errnie.Debug("provider.handleStreamingRequest")
	stream := prvdr.client.Chat.Completions.NewStreaming(prvdr.ctx, *params)
	defer stream.Close()

	acc := openai.ChatCompletionAccumulator{}
	var accumulatedContent string

	for stream.Next() {
		chunk := stream.Current()
		if ok := acc.AddChunk(chunk); !ok {
			errnie.Error("chunk dropped", "id", acc.ID)
			continue
		}
		if content, ok := acc.JustFinishedContent(); ok && content != "" {
			accumulatedContent += content
		}
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			accumulatedContent += chunk.Choices[0].Delta.Content
		}
	}

	if err := stream.Err(); err != nil {
		errnie.Error("Streaming error", "error", err)
		return err
	}

	// Create and append the new assistant message.
	msg, err := NewMessage(prvdr.params.Segment())
	if err != nil {
		return errnie.Error(err)
	}
	if err := msg.SetRole("assistant"); err != nil {
		return errnie.Error(err)
	}
	if prvdr.params.HasModel() {
		model, _ := prvdr.params.Model()
		if err := msg.SetName(model); err != nil {
			return errnie.Error(err)
		}
	}
	if err := msg.SetContent(accumulatedContent); err != nil {
		return errnie.Error(err)
	}

	currentMessages, err := prvdr.params.Messages()
	if err != nil {
		return errnie.Error(err)
	}

	newMessages, err := prvdr.params.NewMessages(int32(currentMessages.Len() + 1))
	if err != nil {
		return errnie.Error(err)
	}
	for i := range currentMessages.Len() {
		if err := newMessages.Set(i, currentMessages.At(i)); err != nil {
			return errnie.Error(err)
		}
	}
	if err := newMessages.Set(currentMessages.Len(), msg); err != nil {
		return errnie.Error(err)
	}
	return prvdr.params.SetMessages(newMessages)
}

func (prvdr *ProviderService) buildMessages(composed *openai.ChatCompletionNewParams) error {
	errnie.Debug("provider.buildMessages")

	if prvdr.params == nil || !prvdr.params.HasMessages() {
		return nil
	}

	messagesList, err := prvdr.params.Messages()

	if err != nil {
		return errnie.Error(err)
	}

	var messages []openai.ChatCompletionMessageParamUnion

	for i := range messagesList.Len() {
		msg := messagesList.At(i)
		converted, err := convertMessage(msg)
		if err != nil {
			return errnie.Error(err)
		}
		messages = append(messages, converted)
	}

	composed.Messages = messages
	return nil
}

func convertMessage(message Message) (openai.ChatCompletionMessageParamUnion, error) {
	role, err := message.Role()

	if err != nil {
		return openai.ChatCompletionMessageParamUnion{}, err
	}

	content, err := message.Content()

	if err != nil {
		return openai.ChatCompletionMessageParamUnion{}, err
	}

	switch role {
	case "system":
		return openai.SystemMessage(content), nil
	case "user":
		return openai.UserMessage(content), nil
	case "assistant":
		if message.HasToolCalls() {
			toolCalls, err := convertToolCalls(message)

			if err != nil {
				return openai.ChatCompletionMessageParamUnion{}, err
			}

			return openai.ChatCompletionMessageParamUnion{
				OfAssistant: &openai.ChatCompletionAssistantMessageParam{
					Content: openai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: param.NewOpt(content),
					},
					ToolCalls: toolCalls,
					Role:      "assistant",
				},
			}, nil
		}
		return openai.AssistantMessage(content), nil
	case "tool":
		ref, err := message.Reference()

		if err != nil {
			return openai.ChatCompletionMessageParamUnion{}, err
		}

		return openai.ChatCompletionMessageParamUnion{
			OfTool: &openai.ChatCompletionToolMessageParam{
				Content: openai.ChatCompletionToolMessageParamContentUnion{
					OfString: param.NewOpt(content),
				},
				ToolCallID: ref,
				Role:       "tool",
			},
		}, nil
	default:
		return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("unknown message role: %s", role)
	}
}

func convertToolCalls(message Message) ([]openai.ChatCompletionMessageToolCallParam, error) {
	toolCallsList, err := message.ToolCalls()

	if err != nil {
		return nil, err
	}

	var toolCalls []openai.ChatCompletionMessageToolCallParam

	for i := range toolCallsList.Len() {
		tc := toolCallsList.At(i)
		id, err := tc.Id()

		if err != nil {
			return nil, err
		}

		function, err := tc.Function()

		if err != nil {
			return nil, err
		}

		name, err := function.Name()

		if err != nil {
			return nil, err
		}

		args, err := function.Arguments()

		if err != nil {
			return nil, err
		}

		toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallParam{
			ID:   id,
			Type: "function",
			Function: openai.ChatCompletionMessageToolCallFunctionParam{
				Name:      name,
				Arguments: args,
			},
		})
	}
	return toolCalls, nil
}

func (prvdr *ProviderService) buildTools(openaiParams *openai.ChatCompletionNewParams) error {
	errnie.Debug("provider.buildTools")
	if openaiParams == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}
	if !prvdr.params.HasTools() {
		return nil
	}
	toolsList, err := prvdr.params.Tools()
	if err != nil {
		return errnie.Error(err)
	}
	if toolsList.Len() == 0 {
		return nil
	}

	var toolsOut []openai.ChatCompletionToolParam
	for i := 0; i < toolsList.Len(); i++ {
		tool := toolsList.At(i)
		function, err := tool.Function()
		if err != nil {
			return errnie.Error(err)
		}
		name, err := function.Name()
		if err != nil {
			return errnie.Error(err)
		}
		description, err := function.Description()
		if err != nil {
			return errnie.Error(err)
		}
		parameters, err := function.Parameters()
		if err != nil {
			return errnie.Error(err)
		}
		properties := make(map[string]any)
		propertiesList, err := parameters.Properties()
		if err != nil {
			return errnie.Error(err)
		}
		for j := 0; j < propertiesList.Len(); j++ {
			property := propertiesList.At(j)
			propName, err := property.Name()
			if err != nil {
				return errnie.Error(err)
			}
			propType, err := property.Type()
			if err != nil {
				return errnie.Error(err)
			}
			propDesc, err := property.Description()
			if err != nil {
				return errnie.Error(err)
			}
			propDef := map[string]any{
				"type":        propType,
				"description": propDesc,
			}
			if property.HasEnum() {
				enumList, err := property.Enum()
				if err != nil {
					return errnie.Error(err)
				}
				var enumValues []string
				for k := 0; k < enumList.Len(); k++ {
					val, err := enumList.At(k)
					if err != nil {
						return errnie.Error(err)
					}
					enumValues = append(enumValues, val)
				}
				if len(enumValues) > 0 {
					propDef["enum"] = enumValues
				}
			}
			properties[propName] = propDef
		}

		parametersSchema := openai.FunctionParameters{
			"type":       "object",
			"properties": properties,
		}
		if parameters.HasRequired() {
			requiredList, err := parameters.Required()
			if err != nil {
				return errnie.Error(err)
			}
			var required []string
			for j := 0; j < requiredList.Len(); j++ {
				val, err := requiredList.At(j)
				if err != nil {
					return errnie.Error(err)
				}
				required = append(required, val)
			}
			parametersSchema["required"] = required
		} else {
			parametersSchema["required"] = []string{}
		}

		toolParam := openai.ChatCompletionToolParam{
			Type: "function",
			Function: openai.FunctionDefinitionParam{
				Name:        name,
				Description: param.NewOpt(description),
				Parameters:  parametersSchema,
			},
		}
		toolsOut = append(toolsOut, toolParam)
	}
	openaiParams.Tools = toolsOut
	return nil
}

func (prvdr *ProviderService) buildResponseFormat(openaiParams *openai.ChatCompletionNewParams) error {
	errnie.Debug("provider.buildResponseFormat")
	if openaiParams == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}
	if !prvdr.params.HasResponseFormat() {
		return nil
	}

	respFormat, err := prvdr.params.ResponseFormat()
	if err != nil {
		return errnie.Error(err)
	}
	name, err := respFormat.Name()
	if err != nil {
		return errnie.Error(err)
	}
	description, err := respFormat.Description()
	if err != nil {
		return errnie.Error(err)
	}
	schema, err := respFormat.Schema()
	if err != nil {
		return errnie.Error(err)
	}

	openaiParams.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			Type: "json_schema",
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        name,
				Description: param.NewOpt(description),
				Schema:      schema,
				Strict:      param.NewOpt(respFormat.Strict()),
			},
		},
	}
	return nil
}

func (prvdr *ProviderService) GetResponse() ([]byte, error) {
	messages, err := prvdr.params.Messages()
	if err != nil {
		return nil, errnie.Error(err)
	}
	if messages.Len() == 0 {
		return nil, errnie.Error(errors.New("no messages in provider params"))
	}
	lastMessage := messages.At(messages.Len() - 1)
	content, err := lastMessage.Content()
	if err != nil {
		return nil, errnie.Error(err)
	}
	return []byte(content), nil
}

// ensureToolResponses ensures that all assistant messages with tool calls have corresponding tool responses
// before sending to the OpenAI API
func (prvdr *ProviderService) ensureToolResponses(messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion {
	var cleanedMessages []openai.ChatCompletionMessageParamUnion
	var pendingToolCalls []struct {
		ID   string
		Name string
	}

	for i, msg := range messages {
		// Add the current message to the cleaned messages
		cleanedMessages = append(cleanedMessages, msg)

		// If this is an assistant message with tool calls
		if msg.OfAssistant != nil && len(msg.OfAssistant.ToolCalls) > 0 {
			// Record all tool calls that need responses
			for _, tc := range msg.OfAssistant.ToolCalls {
				pendingToolCalls = append(pendingToolCalls, struct {
					ID   string
					Name string
				}{
					ID:   tc.ID,
					Name: tc.Function.Name,
				})
			}

			// Check if the next messages are tool responses for these tool calls
			for j := 0; j < len(pendingToolCalls); j++ {
				found := false
				// Look ahead for tool responses
				for k := i + 1; k < len(messages) && k < i+10; k++ {
					if messages[k].OfTool != nil && messages[k].OfTool.ToolCallID == pendingToolCalls[j].ID {
						found = true
						break
					}
				}

				// If no tool response was found for this tool call, add a default one
				if !found {
					errnie.Debug("Adding missing tool response", "id", pendingToolCalls[j].ID, "name", pendingToolCalls[j].Name)
					cleanedMessages = append(cleanedMessages, openai.ChatCompletionMessageParamUnion{
						OfTool: &openai.ChatCompletionToolMessageParam{
							Content: openai.ChatCompletionToolMessageParamContentUnion{
								OfString: param.NewOpt("Tool call processed"),
							},
							ToolCallID: pendingToolCalls[j].ID,
							Role:       "tool",
						},
					})
				}
			}

			// Clear pending tool calls after processing this assistant message
			pendingToolCalls = nil
		}
	}

	return cleanedMessages
}
