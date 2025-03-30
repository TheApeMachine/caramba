package provider

import (
	"context"
	"errors"
	"fmt"

	capnp "capnproto.org/go/capnp/v3"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
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

// Complete handles a single completion request
func (p *CapnpProvider) Complete(ctx context.Context, call Provider_complete) error {
	params := call.Args()
	provParams, err := params.Params()
	if err != nil {
		return errnie.Error(err)
	}

	// Convert Cap'n Proto params to OpenAI format
	model, err := provParams.Model()
	if err != nil {
		return errnie.Error(err)
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

	// Convert messages
	messages := make([]openai.ChatCompletionMessageParamUnion, 0)
	messageList, err := provParams.Messages()
	if err != nil {
		return errnie.Error(err)
	}

	for i := range messageList.Len() {
		msg := messageList.At(i)
		role, err := msg.Role()
		if err != nil {
			return errnie.Error(err)
		}
		content, err := msg.Content()
		if err != nil {
			return errnie.Error(err)
		}

		switch role {
		case "system":
			messages = append(messages, openai.SystemMessage(content))
		case "user":
			messages = append(messages, openai.UserMessage(content))
		case "assistant":
			messages = append(messages, openai.AssistantMessage(content))
		default:
			return errnie.Error(fmt.Errorf("unknown message role: %s", role))
		}
	}
	chatParams.Messages = messages

	// Make the API call
	completion, err := p.client.Chat.Completions.New(p.ctx, *chatParams)
	if err != nil {
		return errnie.Error(err)
	}

	// Set the response
	results, err := call.AllocResults()
	if err != nil {
		return errnie.Error(err)
	}

	// Copy parameters to results
	if err := CopyProviderParams(&provParams, &results); err != nil {
		return errnie.Error(err)
	}

	// Get existing messages
	existingMessages, err := results.Messages()
	if err != nil {
		return errnie.Error(err)
	}

	// Create new message list with length + 1
	resultMessages, err := results.NewMessages(int32(existingMessages.Len() + 1))
	if err != nil {
		return errnie.Error(err)
	}

	// Copy existing messages
	for i := 0; i < existingMessages.Len(); i++ {
		resultMessages.Set(i, existingMessages.At(i))
	}

	// Add the new assistant message
	resultMsg := resultMessages.At(existingMessages.Len())
	if err := resultMsg.SetRole("assistant"); err != nil {
		return errnie.Error(err)
	}
	if err := resultMsg.SetContent(completion.Choices[0].Message.Content); err != nil {
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

	// Convert Cap'n Proto params to OpenAI format
	model, err := provParams.Model()
	if err != nil {
		return errnie.Error(err)
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

	// Convert messages
	messages := make([]openai.ChatCompletionMessageParamUnion, 0)
	messageList, err := provParams.Messages()
	if err != nil {
		return errnie.Error(err)
	}

	for i := 0; i < messageList.Len(); i++ {
		msg := messageList.At(i)
		role, err := msg.Role()
		if err != nil {
			return errnie.Error(err)
		}
		content, err := msg.Content()
		if err != nil {
			return errnie.Error(err)
		}

		switch role {
		case "system":
			messages = append(messages, openai.SystemMessage(content))
		case "user":
			messages = append(messages, openai.UserMessage(content))
		case "assistant":
			messages = append(messages, openai.AssistantMessage(content))
		default:
			return errnie.Error(fmt.Errorf("unknown message role: %s", role))
		}
	}
	chatParams.Messages = messages

	// Create streaming client
	stream := p.client.Chat.Completions.NewStreaming(p.ctx, *chatParams)
	defer stream.Close()

	// Get result params ready
	results, err := call.AllocResults()
	if err != nil {
		return errnie.Error(err)
	}

	// Copy basic params
	if err := results.SetModel(model); err != nil {
		return errnie.Error(err)
	}
	results.SetTemperature(provParams.Temperature())
	results.SetTopP(provParams.TopP())
	results.SetFrequencyPenalty(provParams.FrequencyPenalty())
	results.SetPresencePenalty(provParams.PresencePenalty())
	results.SetMaxTokens(provParams.MaxTokens())

	// Stream responses
	acc := openai.ChatCompletionAccumulator{}
	for stream.Next() {
		chunk := stream.Current()
		if ok := acc.AddChunk(chunk); !ok {
			errnie.Error("chunk dropped", "id", acc.ID)
			continue
		}

		if content, ok := acc.JustFinishedContent(); ok && content != "" {
			// Add the response message
			resultMessages, err := results.NewMessages(1)
			if err != nil {
				return errnie.Error(err)
			}
			resultMsg := resultMessages.At(0)
			if err := resultMsg.SetRole("assistant"); err != nil {
				return errnie.Error(err)
			}
			if err := resultMsg.SetContent(content); err != nil {
				return errnie.Error(err)
			}
		}
	}

	if err := stream.Err(); err != nil {
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

// NewUserMessage creates a new message with the given content
func NewUserMessage(seg *capnp.Segment, content string) (*Message, error) {
	msg, err := NewMessage(seg)
	if err != nil {
		return nil, errnie.Error(err)
	}

	if err := msg.SetRole("user"); err != nil {
		return nil, errnie.Error(err)
	}
	if err := msg.SetContent(content); err != nil {
		return nil, errnie.Error(err)
	}

	return &msg, nil
}

// NewRequest creates a new request with the given user message
func NewRequest(seg *capnp.Segment, content string) (*ProviderParams, error) {
	// Create a new root struct for the params
	params, err := NewRootProviderParams(seg)
	if err != nil {
		return nil, errnie.Error(err)
	}

	// Set default model
	if err := params.SetModel(tweaker.GetModel("openai")); err != nil {
		return nil, errnie.Error(err)
	}

	// Create a new message
	msg, err := NewUserMessage(seg, content)
	if err != nil {
		return nil, errnie.Error(err)
	}

	// Create a new message list with length 1
	messages, err := params.NewMessages(1)
	if err != nil {
		return nil, errnie.Error(err)
	}

	// Set the message in the list
	messages.Set(0, *msg)

	return &params, nil
}

// NewConversation creates a new conversation with an initial user message
func NewConversation(content string) (*ProviderParams, error) {
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

	// Create and set the user message
	msg, err := NewUserMessage(seg, content)
	if err != nil {
		return nil, errnie.Error(err)
	}

	// Create a new message list
	messages, err := params.NewMessages(1)
	if err != nil {
		return nil, errnie.Error(err)
	}
	messages.Set(0, *msg)

	return &params, nil
}

// AddUserMessage adds a new user message to existing conversation params
func AddUserMessage(params *ProviderParams, content string) error {
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
	msg, err := NewUserMessage(params.Segment(), content)
	if err != nil {
		return errnie.Error(err)
	}
	newMessages.Set(currentMessages.Len(), *msg)

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
