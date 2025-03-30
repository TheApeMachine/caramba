package provider

import (
	"context"
	"errors"
	"fmt"

	capnp "capnproto.org/go/capnp/v3"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

// CapnpProvider implements the Cap'n Proto Provider interface
type CapnpProvider struct {
	client *openai.Client
	ctx    context.Context
	cancel context.CancelFunc
}

// NewCapnpProvider creates a new Cap'n Proto provider with the given API key
func NewCapnpProvider(apiKey string) *CapnpProvider {
	ctx, cancel := context.WithCancel(context.Background())
	client := openai.NewClient(option.WithAPIKey(apiKey))

	return &CapnpProvider{
		client: &client,
		ctx:    ctx,
		cancel: cancel,
	}
}

// convertProviderParamsToOpenAI converts Cap'n Proto params to OpenAI format
func convertProviderParamsToOpenAI(provParams ProviderParams) (*openai.ChatCompletionNewParams, error) {
	model, err := provParams.Model()
	if err != nil {
		return nil, errnie.Error(err)
	}

	chatParams := &openai.ChatCompletionNewParams{
		Model:            model,
		Temperature:      openai.Float(provParams.Temperature()),
		TopP:             openai.Float(provParams.TopP()),
		FrequencyPenalty: openai.Float(provParams.FrequencyPenalty()),
		PresencePenalty:  openai.Float(provParams.PresencePenalty()),
	}

	if provParams.MaxTokens() > 0 {
		chatParams.MaxTokens = openai.Int(int64(provParams.MaxTokens()))
	}

	messages, err := convertMessagesToOpenAI(provParams)
	if err != nil {
		return nil, err
	}
	chatParams.Messages = messages

	return chatParams, nil
}

// convertMessagesToOpenAI converts Cap'n Proto messages to OpenAI format
func convertMessagesToOpenAI(provParams ProviderParams) ([]openai.ChatCompletionMessageParamUnion, error) {
	messages := make([]openai.ChatCompletionMessageParamUnion, 0)
	messageList, err := provParams.Messages()
	if err != nil {
		return nil, errnie.Error(err)
	}

	for i := 0; i < messageList.Len(); i++ {
		msg := messageList.At(i)
		role, err := msg.Role()
		if err != nil {
			return nil, errnie.Error(err)
		}
		content, err := msg.Content()
		if err != nil {
			return nil, errnie.Error(err)
		}

		switch role {
		case "system":
			messages = append(messages, openai.SystemMessage(content))
		case "user":
			messages = append(messages, openai.UserMessage(content))
		case "assistant":
			toolcalls, err := msg.ToolCalls()
			if err != nil {
				return nil, errnie.Error(err)
			}
			toolCalls := make([]openai.ChatCompletionMessageToolCallParam, 0, toolcalls.Len())

			for i := range toolcalls.Len() {
				toolCall := toolcalls.At(i)

				id, err := toolCall.Id()
				if err != nil {
					return nil, errnie.Error(err)
				}

				function, err := toolCall.Function()
				if err != nil {
					return nil, errnie.Error(err)
				}

				name, err := function.Name()
				if err != nil {
					return nil, errnie.Error(err)
				}

				arguments, err := function.Arguments()
				if err != nil {
					return nil, errnie.Error(err)
				}

				toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallParam{
					ID:   id,
					Type: "function",
					Function: openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      name,
						Arguments: arguments,
					},
				})
			}

			errnie.Info("toolCalls", "toolCalls", toolCalls)

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
			reference, err := msg.Reference()
			if err != nil {
				return nil, errnie.Error(err)
			}
			messages = append(messages, openai.ChatCompletionMessageParamUnion{
				OfTool: &openai.ChatCompletionToolMessageParam{
					Content: openai.ChatCompletionToolMessageParamContentUnion{
						OfString: param.NewOpt(content),
					},
					ToolCallID: reference,
					Role:       "tool",
				},
			})
		default:
			return nil, errnie.Error(fmt.Errorf("unknown message role: %s", role))
		}
	}

	return messages, nil
}

// Complete handles a single completion request
func (p *CapnpProvider) Complete(ctx context.Context, call Provider_complete) error {
	params := call.Args()
	provParams, err := params.Params()
	if err != nil {
		return errnie.Error(err)
	}

	chatParams, err := convertProviderParamsToOpenAI(provParams)
	if err != nil {
		return errnie.Error(err)
	}

	completion, err := p.client.Chat.Completions.New(p.ctx, *chatParams)
	if err != nil {
		return errnie.Error(err)
	}

	results, err := call.AllocResults()
	if err != nil {
		return errnie.Error(err)
	}

	if err := CopyProviderParams(&provParams, &results); err != nil {
		return errnie.Error(err)
	}

	if err := appendMessageToResults(&results, "assistant", completion.Choices[0].Message.Content); err != nil {
		return errnie.Error(err)
	}

	return nil
}

// Stream handles a streaming completion request
func (p *CapnpProvider) Stream(ctx context.Context, call Provider_stream) error {
	params := call.Args()
	provParams, err := params.Params()
	if err != nil {
		return errnie.Error(err)
	}

	chatParams, err := convertProviderParamsToOpenAI(provParams)
	if err != nil {
		return errnie.Error(err)
	}

	stream := p.client.Chat.Completions.NewStreaming(p.ctx, *chatParams)
	defer stream.Close()

	results, err := call.AllocResults()
	if err != nil {
		return errnie.Error(err)
	}

	if err := CopyProviderParams(&provParams, &results); err != nil {
		return errnie.Error(err)
	}

	acc := openai.ChatCompletionAccumulator{}
	for stream.Next() {
		chunk := stream.Current()
		if ok := acc.AddChunk(chunk); !ok {
			errnie.Error("chunk dropped", "id", acc.ID)
			continue
		}

		if content, ok := acc.JustFinishedContent(); ok && content != "" {
			if err := appendMessageToResults(&results, "assistant", content); err != nil {
				return errnie.Error(err)
			}
		}
	}

	if err := stream.Err(); err != nil {
		return errnie.Error(err)
	}

	return nil
}

// appendMessageToResults adds a new message to the results
func appendMessageToResults(results *ProviderParams, role, content string) error {
	messages, err := results.NewMessages(1)
	if err != nil {
		return errnie.Error(err)
	}
	msg := messages.At(0)
	if err := msg.SetRole(role); err != nil {
		return errnie.Error(err)
	}
	if err := msg.SetContent(content); err != nil {
		return errnie.Error(err)
	}
	return nil
}

// Embed creates embeddings for the given text
func (p *CapnpProvider) Embed(ctx context.Context, call Provider_embed) error {
	params := call.Args()
	text, err := params.Text()
	if err != nil {
		return errnie.Error(err)
	}

	// Create embedding request
	response, err := p.client.Embeddings.New(p.ctx, openai.EmbeddingNewParams{
		Input:          openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: []string{text}},
		Model:          openai.EmbeddingModelTextEmbeddingAda002,
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	})
	if err != nil {
		return errnie.Error(err)
	}

	if len(response.Data) == 0 {
		return errnie.Error(errors.New("no embeddings returned"))
	}

	// Get result params
	results, err := call.AllocResults()
	if err != nil {
		return errnie.Error(err)
	}

	// Convert float64 embeddings to float32 and store in result
	embedding, err := results.NewEmbedding(int32(len(response.Data[0].Embedding)))
	if err != nil {
		return errnie.Error(err)
	}

	for i, v := range response.Data[0].Embedding {
		embedding.Set(i, float32(v))
	}

	return nil
}

// CopyProviderParams copies values from src to dst ProviderParams
func CopyProviderParams(src, dst *ProviderParams) error {
	model, err := src.Model()
	if err != nil || model == "" {
		// Use default model if not set
		model = tweaker.GetModel("openai")
	}
	if err = dst.SetModel(model); err != nil {
		return errnie.Error(err)
	}
	dst.SetTemperature(src.Temperature())
	dst.SetTopP(src.TopP())
	dst.SetFrequencyPenalty(src.FrequencyPenalty())
	dst.SetPresencePenalty(src.PresencePenalty())
	dst.SetMaxTokens(src.MaxTokens())

	// Copy messages
	srcMessages, err := src.Messages()
	if err != nil {
		return errnie.Error(err)
	}

	if srcMessages.Len() > 0 {
		dstMessages, err := dst.NewMessages(int32(srcMessages.Len()))
		if err != nil {
			return errnie.Error(err)
		}

		for i := 0; i < srcMessages.Len(); i++ {
			srcMsg := srcMessages.At(i)
			dstMsg := dstMessages.At(i)

			// Copy role
			role, err := srcMsg.Role()
			if err != nil {
				return errnie.Error(err)
			}
			if err := dstMsg.SetRole(role); err != nil {
				return errnie.Error(err)
			}

			// Copy content
			content, err := srcMsg.Content()
			if err != nil {
				return errnie.Error(err)
			}
			if err := dstMsg.SetContent(content); err != nil {
				return errnie.Error(err)
			}
		}
	}

	return nil
}

// NewProvider creates a new provider with the given API key and returns it as a Provider interface
func NewProvider(apiKey string) Provider {
	return Provider_ServerToClient(NewCapnpProvider(apiKey))
}

// NewConversation creates a new conversation with an initial user message
func NewConversation() (*ProviderParams, error) {
	// Create a new message and segment internally
	_, seg, err := capnp.NewMessage(capnp.SingleSegment(nil))
	if err != nil {
		return nil, errnie.Error(err)
	}

	// Create root params
	params, err := NewRootProviderParams(seg)
	if err != nil {
		return nil, errnie.Error(err)
	}

	// Set the model
	if err := params.SetModel(tweaker.GetModel("openai")); err != nil {
		return nil, errnie.Error(err)
	}

	return &params, nil
}

// addMessage is a helper function that adds a new message to the conversation
func addMessage(params *ProviderParams, role, name, content string) error {
	// Get current messages
	currentMessages, err := params.Messages()
	if err != nil {
		return errnie.Error(err)
	}

	// Create new message list with length + 1
	newMessages, err := params.NewMessages(int32(currentMessages.Len() + 1))
	if err != nil {
		return errnie.Error(err)
	}

	// Copy existing messages
	for i := 0; i < currentMessages.Len(); i++ {
		newMessages.Set(i, currentMessages.At(i))
	}

	// Add the new message
	msg, err := NewMessage(params.Segment())
	if err != nil {
		return errnie.Error(err)
	}

	if err := msg.SetRole(role); err != nil {
		return errnie.Error(err)
	}

	if name != "" {
		if err := msg.SetName(name); err != nil {
			return errnie.Error(err)
		}
	}

	if err := msg.SetContent(content); err != nil {
		return errnie.Error(err)
	}

	newMessages.Set(currentMessages.Len(), msg)

	return nil
}

// AddSystemMessage adds a new system message to existing conversation params
func AddSystemMessage(params *ProviderParams, content string) error {
	return addMessage(params, "system", "", content)
}

// AddUserMessage adds a new user message to existing conversation params
func AddUserMessage(params *ProviderParams, name, content string) error {
	return addMessage(params, "user", name, content)
}

// AddAssistantMessage adds a new assistant message to existing conversation params
func AddAssistantMessage(params *ProviderParams, content string) error {
	return addMessage(params, "assistant", "", content)
}

// AddTool appends a new tool to the provider params Tools list
func AddTool(params *ProviderParams, toolName string) error {
	currentTools, err := params.Tools()
	if err != nil {
		return errnie.Error(err)
	}

	tools, err := params.NewTools(int32(currentTools.Len() + 1))
	if err != nil {
		return errnie.Error(err)
	}

	tool, err := NewTool(params.Segment())
	if err != nil {
		return errnie.Error(err)
	}

	function, err := NewFunction(params.Segment())
	if err != nil {
		return errnie.Error(err)
	}

	parameters, err := NewParameters(params.Segment())
	if err != nil {
		return errnie.Error(err)
	}

	base := fmt.Sprintf("tools.schemas.%s", toolName)

	// Get function details from config
	name := tweaker.Get(base+".function.name", toolName)
	description := tweaker.Get(base+".function.description", "")

	if err := function.SetName(name); err != nil {
		return errnie.Error(err)
	}

	if err := function.SetDescription(description); err != nil {
		return errnie.Error(err)
	}

	if err := parameters.SetType("object"); err != nil {
		return errnie.Error(err)
	}

	// Get properties from config
	if props := tweaker.GetStringMap(base + ".properties"); len(props) > 0 {
		properties, err := parameters.NewProperties(int32(len(props)))
		if err != nil {
			return errnie.Error(err)
		}

		i := 0
		for propName := range props {
			propBase := fmt.Sprintf("%s.properties.%s", base, propName)

			property, err := NewProperty(params.Segment())
			if err != nil {
				return errnie.Error(err)
			}

			if err := property.SetName(propName); err != nil {
				return errnie.Error(err)
			}

			if err := property.SetType(tweaker.Get(propBase+".type", "string")); err != nil {
				return errnie.Error(err)
			}

			if err := property.SetDescription(tweaker.Get(propBase+".description", "")); err != nil {
				return errnie.Error(err)
			}

			// Handle enum/options if present
			if options := tweaker.GetStringSlice(propBase + ".options"); len(options) > 0 {
				enum, err := property.NewEnum(int32(len(options)))
				if err != nil {
					return errnie.Error(err)
				}
				for j, opt := range options {
					enum.Set(j, opt)
				}
			}

			properties.Set(i, property)
			i++
		}
	}

	// Handle required fields if present
	if required := tweaker.GetStringSlice(base + ".required"); len(required) > 0 {
		requiredList, err := parameters.NewRequired(int32(len(required)))
		if err != nil {
			return errnie.Error(err)
		}
		for i, req := range required {
			requiredList.Set(i, req)
		}
	}

	if err := function.SetParameters(parameters); err != nil {
		return errnie.Error(err)
	}

	if err := tool.SetFunction(function); err != nil {
		return errnie.Error(err)
	}

	tools.Set(currentTools.Len(), tool)

	return nil
}

// GetLastMessageContent returns the content of the last message in the conversation
func GetLastMessage(params *ProviderParams) (*Message, error) {
	messages, err := params.Messages()
	if err != nil {
		return nil, errnie.Error(err)
	}
	if messages.Len() == 0 {
		return nil, errnie.Error(fmt.Errorf("no messages in conversation"))
	}
	lastMsg := messages.At(messages.Len() - 1)

	return &lastMsg, nil
}
