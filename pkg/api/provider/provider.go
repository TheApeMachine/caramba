package provider

import (
	context "context"
	"errors"
	"fmt"
	"os"

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
	client := openai.NewClient(
		option.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	)

	return &ProviderService{
		client: &client,
		params: &ProviderParams{},
		ctx:    context.Background(),
	}
}

func (prvdr *ProviderService) SetParams(params *ProviderParams) {
	prvdr.params = params
}

func (prvdr *ProviderService) Generate() (err error) {
	errnie.Debug("provider.Generate")

	model, err := prvdr.params.Model()

	if err != nil {
		return errnie.Error(err)
	}

	temperature := prvdr.params.Temperature()
	topP := prvdr.params.TopP()
	frequencyPenalty := prvdr.params.FrequencyPenalty()
	presencePenalty := prvdr.params.PresencePenalty()
	maxTokens := prvdr.params.MaxTokens()

	composed := &openai.ChatCompletionNewParams{
		Model:            model,
		Temperature:      openai.Float(temperature),
		TopP:             openai.Float(topP),
		FrequencyPenalty: openai.Float(frequencyPenalty),
		PresencePenalty:  openai.Float(presencePenalty),
	}

	if maxTokens > 1 {
		composed.MaxTokens = openai.Int(int64(maxTokens))
	}

	if err = prvdr.buildMessages(composed); err != nil {
		return errnie.Error(err)
	}

	if err = prvdr.buildTools(composed); err != nil {
		return errnie.Error(err)
	}

	if prvdr.params.HasResponseFormat() {
		if err = prvdr.buildResponseFormat(composed); err != nil {
			return errnie.Error(err)
		}
	}

	return prvdr.handleSingleRequest(composed)
}

/*
handleSingleRequest processes a single (non-streaming) completion request
*/
func (prvdr *ProviderService) handleSingleRequest(
	params *openai.ChatCompletionNewParams,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	var completion *openai.ChatCompletion

	for _, message := range params.Messages {
		errnie.Debug("message", "message", message)
	}

	if completion, err = prvdr.client.Chat.Completions.New(
		context.Background(), *params,
	); errnie.Error(err) != nil {
		return err
	}

	messages, err := prvdr.params.Messages()

	if err != nil {
		return errnie.Error(err)
	}

	msgList, err := NewMessage_List(
		prvdr.params.Segment(),
		int32(messages.Len()+1),
	)

	if err != nil {
		return errnie.Error(err)
	}

	for i := range messages.Len() {
		msg := messages.At(i)
		msgList.Set(i, msg)
	}

	toolCalls := completion.Choices[0].Message.ToolCalls

	// Handle tool calls if they exist
	if len(toolCalls) > 0 {
		params.Messages = append(params.Messages, completion.Choices[0].Message.ToParam())

		toolCallsList, err := NewToolCall_List(prvdr.params.Segment(), int32(len(toolCalls)))

		if err != nil {
			return errnie.Error(err)
		}

		for i, toolCall := range toolCalls {
			errnie.Debug("toolCall", "tool", toolCall.Function.Name, "id", toolCall.ID)

			tc := toolCallsList.At(i)

			if err = tc.SetId(toolCall.ID); err != nil {
				return errnie.Error(err)
			}

			if err = tc.SetType("function"); err != nil {
				return errnie.Error(err)
			}

			tcFunc, err := tc.NewFunction()

			if err != nil {
				return errnie.Error(err)
			}

			if err = tcFunc.SetName(toolCall.Function.Name); err != nil {
				return errnie.Error(err)
			}

			if err = tcFunc.SetArguments(toolCall.Function.Arguments); err != nil {
				return errnie.Error(err)
			}
		}

		msg, err := NewMessage(prvdr.params.Segment())

		if err != nil {
			return errnie.Error(err)
		}

		msg.SetRole("assistant")
		msg.SetToolCalls(toolCallsList)
		msg.SetContent(completion.Choices[0].Message.Content)

		msgList.Set(messages.Len(), msg)

		if err = prvdr.params.SetMessages(msgList); err != nil {
			return errnie.Error(err)
		}

		return nil
	}

	msg, err := NewMessage(prvdr.params.Segment())

	if err != nil {
		return errnie.Error(err)
	}

	msg.SetRole("assistant")
	msg.SetContent(completion.Choices[0].Message.Content)

	msgList.Set(messages.Len(), msg)

	if err = prvdr.params.SetMessages(msgList); err != nil {
		return errnie.Error(err)
	}

	return nil
}

/*
handleStreamingRequest processes a streaming completion request
and accumulates the response to add to the context
*/
func (prvdr *ProviderService) handleStreamingRequest(
	params *openai.ChatCompletionNewParams,
) (err error) {
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

		// Only add non-empty content from chunks
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			accumulatedContent += chunk.Choices[0].Delta.Content
		}
	}

	if err = stream.Err(); err != nil {
		errnie.Error("Streaming error", "error", err)
		return err
	}

	// Create a new assistant message with the accumulated content
	msg, err := NewMessage(prvdr.params.Segment())
	if err != nil {
		return errnie.Error(err)
	}

	if err = msg.SetRole("assistant"); err != nil {
		return errnie.Error(err)
	}

	if prvdr.params.HasModel() {
		model, _ := prvdr.params.Model()
		if err = msg.SetName(model); err != nil {
			return errnie.Error(err)
		}
	}

	if err = msg.SetContent(accumulatedContent); err != nil {
		return errnie.Error(err)
	}

	// Add the new message to the existing messages in params
	currentMessages, err := prvdr.params.Messages()
	if err != nil {
		return errnie.Error(err)
	}

	// Create a new message list with one more slot for our new message
	newMessages, err := prvdr.params.NewMessages(int32(currentMessages.Len() + 1))
	if err != nil {
		return errnie.Error(err)
	}

	// Copy over existing messages
	for i := range currentMessages.Len() {
		if err = newMessages.Set(i, currentMessages.At(i)); err != nil {
			return errnie.Error(err)
		}
	}

	// Add the new message at the end
	if err = newMessages.Set(currentMessages.Len(), msg); err != nil {
		return errnie.Error(err)
	}

	// Update the messages in the params
	if err = prvdr.params.SetMessages(newMessages); err != nil {
		return errnie.Error(err)
	}

	return nil
}

/*
buildMessages converts ProviderParams messages to OpenAI API format
*/
func (prvdr *ProviderService) buildMessages(
	composed *openai.ChatCompletionNewParams,
) (err error) {
	errnie.Debug("provider.buildMessages")

	if prvdr.params == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	if !prvdr.params.HasMessages() {
		return nil // No messages, no problem
	}

	messagesList, err := prvdr.params.Messages()
	if err != nil {
		return errnie.Error(err)
	}

	messagesLen := messagesList.Len()
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, messagesLen)

	for i := range messagesLen {
		message := messagesList.At(i)

		role, err := message.Role()
		if err != nil {
			return errnie.Error(err)
		}

		content, err := message.Content()
		if err != nil {
			return errnie.Error(err)
		}

		switch role {
		case "system":
			messages = append(messages, openai.SystemMessage(content))
		case "user":
			messages = append(messages, openai.UserMessage(content))
		case "assistant":
			var toolCalls []openai.ChatCompletionMessageToolCallParam

			if message.HasToolCalls() {
				// Process tool calls if present
				toolCallsList, err := message.ToolCalls()
				if err != nil {
					return errnie.Error(err)
				}

				toolCallsLen := toolCallsList.Len()
				toolCalls = make([]openai.ChatCompletionMessageToolCallParam, 0, toolCallsLen)

				for j := range toolCallsLen {
					toolCall := toolCallsList.At(j)

					id, err := toolCall.Id()
					if err != nil {
						return errnie.Error(err)
					}

					function, err := toolCall.Function()
					if err != nil {
						return errnie.Error(err)
					}

					name, err := function.Name()
					if err != nil {
						return errnie.Error(err)
					}

					args, err := function.Arguments()
					if err != nil {
						return errnie.Error(err)
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
			}

			msg := openai.AssistantMessage(content)
			if len(toolCalls) > 0 {
				msg = openai.ChatCompletionMessageParamUnion{
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						Content: openai.ChatCompletionAssistantMessageParamContentUnion{
							OfString: param.NewOpt(content),
						},
						ToolCalls: toolCalls,
						Role:      "assistant",
					},
				}
			}

			messages = append(messages, msg)
		case "tool":
			ref, err := message.Reference()
			if err != nil {
				return errnie.Error(err)
			}

			messages = append(messages, openai.ChatCompletionMessageParamUnion{
				OfTool: &openai.ChatCompletionToolMessageParam{
					Content: openai.ChatCompletionToolMessageParamContentUnion{
						OfString: param.NewOpt(content),
					},
					ToolCallID: ref,
					Role:       "tool",
				},
			})
		default:
			fmt.Println(role, content)
			return errnie.Error(errors.New("unknown message role"))
		}
	}

	composed.Messages = messages

	return nil
}

/*
buildTools takes the tools from the generic params and converts them to OpenAI API format.
It is important to return nil early when there are no tools, because passing an empty array
to the OpenAI API will cause strange behavior, like the model guessing random tools.
*/
func (prvdr *ProviderService) buildTools(
	openaiParams *openai.ChatCompletionNewParams,
) (err error) {
	errnie.Debug("provider.buildTools")

	if openaiParams == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	if !prvdr.params.HasTools() {
		// No tools, no shoes, no dice.
		return nil
	}

	toolsList, err := prvdr.params.Tools()
	if err != nil {
		return errnie.Error(err)
	}

	toolsLen := toolsList.Len()
	if toolsLen == 0 {
		return nil
	}

	toolsOut := make([]openai.ChatCompletionToolParam, 0, toolsLen)

	for i := 0; i < toolsLen; i++ {
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

				enumValues := make([]string, 0, enumList.Len())
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

		// Ensure we always have a valid parameters object that matches OpenAI's schema
		parametersSchema := openai.FunctionParameters{
			"type":       "object",
			"properties": properties,
		}

		// Only include required if it has values
		if parameters.HasRequired() {
			requiredList, err := parameters.Required()
			if err != nil {
				return errnie.Error(err)
			}

			required := make([]string, 0, requiredList.Len())
			for j := 0; j < requiredList.Len(); j++ {
				val, err := requiredList.At(j)
				if err != nil {
					return errnie.Error(err)
				}
				required = append(required, val)
			}

			if len(required) > 0 {
				parametersSchema["required"] = required
			} else {
				parametersSchema["required"] = []string{}
			}
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

/*
buildResponseFormat converts the response format from the generic params to OpenAI API format.
This will force the model to use structured output, and return a JSON object.
Setting Strict to true will make sure the only thing returned is the JSON object.
If you want this to be combined with the ability to call tools, you can set Strict to false.
*/
func (prvdr *ProviderService) buildResponseFormat(
	openaiParams *openai.ChatCompletionNewParams,
) (err error) {
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

// GetResponse returns the response data from the provider
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
