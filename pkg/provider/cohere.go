package provider

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"

	cohere "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

/*
CohereProvider implements an LLM provider that connects to Cohere's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type CohereProvider struct {
	client   *cohereclient.Client
	endpoint string
	buffer   *stream.Buffer
	params   *Params
	ctx      context.Context
	cancel   context.CancelFunc
}

/*
NewCohereProvider creates a new Cohere provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the COHERE_API_KEY environment variable.
*/
func NewCohereProvider(opts ...CohereProviderOption) *CohereProvider {
	errnie.Debug("provider.NewCohereProvider")

	apiKey := os.Getenv("COHERE_API_KEY")
	endpoint := viper.GetViper().GetString("endpoints.cohere")
	ctx, cancel := context.WithCancel(context.Background())

	prvdr := &CohereProvider{
		client: cohereclient.NewClient(
			cohereclient.WithToken(apiKey),
		),
		endpoint: endpoint,
		buffer:   stream.NewBuffer(),
		params:   &Params{},
		ctx:      ctx,
		cancel:   cancel,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *CohereProvider) ID() string {
	return "cohere"
}

type CohereProviderOption func(*CohereProvider)

func WithCohereAPIKey(apiKey string) CohereProviderOption {
	return func(prvdr *CohereProvider) {
		prvdr.client = cohereclient.NewClient(
			cohereclient.WithToken(apiKey),
		)
	}
}

func WithCohereEndpoint(endpoint string) CohereProviderOption {
	return func(prvdr *CohereProvider) {
		prvdr.endpoint = endpoint
	}
}

func (prvdr *CohereProvider) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("provider.CohereProvider.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		select {
		case <-prvdr.ctx.Done():
			errnie.Debug("provider.CohereProvider.Generate.ctx.Done")
			prvdr.cancel()
			return
		case artifact := <-buffer:
			if err := artifact.To(prvdr.params); err != nil {
				out <- datura.New(datura.WithError(errnie.Error(err)))
				return
			}

			composed := &cohere.ChatStreamRequest{
				Model:            cohere.String(prvdr.params.Model),
				Temperature:      cohere.Float64(prvdr.params.Temperature),
				P:                cohere.Float64(prvdr.params.TopP),
				K:                cohere.Int(int(prvdr.params.TopK)),
				PresencePenalty:  cohere.Float64(prvdr.params.PresencePenalty),
				FrequencyPenalty: cohere.Float64(prvdr.params.FrequencyPenalty),
				MaxTokens:        cohere.Int(int(prvdr.params.MaxTokens)),
			}

			if err := prvdr.buildMessages(composed); err != nil {
				out <- datura.New(datura.WithError(err))
				return
			}

			if err := prvdr.buildTools(composed); err != nil {
				out <- datura.New(datura.WithError(err))
				return
			}

			if prvdr.params.ResponseFormat != nil {
				if err := prvdr.buildResponseFormat(composed); err != nil {
					out <- datura.New(datura.WithError(err))
					return
				}
			}

			var err error
			if prvdr.params.Stream {
				err = prvdr.handleStreamingRequest(composed)
			} else {
				err = prvdr.handleSingleRequest(composed)
			}

			if err != nil {
				out <- datura.New(datura.WithError(err))
				return
			}

			out <- datura.New(datura.WithPayload(prvdr.params.Marshal()))
		}
	}()

	return out
}

func (prvdr *CohereProvider) Name() string {
	return "cohere"
}

/*
Read implements the io.Reader interface.
*/
func (prvdr *CohereProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.CohereProvider.Read")
	return prvdr.buffer.Read(p)
}

/*
Write implements the io.Writer interface.
*/
func (prvdr *CohereProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.CohereProvider.Write")

	if n, err = prvdr.buffer.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	composed := &cohere.ChatStreamRequest{
		Model:            cohere.String(prvdr.params.Model),
		Temperature:      cohere.Float64(prvdr.params.Temperature),
		P:                cohere.Float64(prvdr.params.TopP),
		K:                cohere.Int(int(prvdr.params.TopK)),
		PresencePenalty:  cohere.Float64(prvdr.params.PresencePenalty),
		FrequencyPenalty: cohere.Float64(prvdr.params.FrequencyPenalty),
		MaxTokens:        cohere.Int(int(prvdr.params.MaxTokens)),
	}

	prvdr.buildMessages(composed)
	prvdr.buildTools(composed)
	prvdr.buildResponseFormat(composed)

	if prvdr.params.Stream {
		prvdr.handleStreamingRequest(composed)
	} else {
		prvdr.handleSingleRequest(composed)
	}

	return n, nil
}

/*
Close cleans up any resources.
*/
func (prvdr *CohereProvider) Close() error {
	errnie.Debug("provider.CohereProvider.Close")
	prvdr.cancel()
	return nil
}

func (prvdr *CohereProvider) handleSingleRequest(
	params *cohere.ChatStreamRequest,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	// Convert stream request to regular chat request
	chatRequest := &cohere.ChatRequest{
		Model:       params.Model,
		Message:     params.Message,
		ChatHistory: params.ChatHistory,
		Preamble:    params.Preamble,
		Tools:       params.Tools,
		Temperature: params.Temperature,
	}

	response, err := prvdr.client.Chat(prvdr.ctx, chatRequest)
	if errnie.Error(err) != nil {
		return
	}

	if _, err = io.Copy(prvdr, datura.New(
		datura.WithPayload([]byte(response.Text)),
	)); errnie.Error(err) != nil {
		return err
	}

	return nil
}

func (prvdr *CohereProvider) handleStreamingRequest(
	params *cohere.ChatStreamRequest,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	stream, err := prvdr.client.ChatStream(prvdr.ctx, params)

	if errnie.Error(err) != nil {
		return

	}

	defer stream.Close()

	for {
		chunk, err := stream.Recv()

		if err != nil {
			if err == io.EOF {
				break
			}

			return errnie.Error(err)
		}

		if content := chunk.TextGeneration.String(); content != "" {
			if _, err = io.Copy(prvdr, datura.New(
				datura.WithPayload([]byte(content)),
			)); errnie.Error(err) != nil {
				continue
			}
		}
	}

	return nil
}

func (prvdr *CohereProvider) buildMessages(
	chatParams *cohere.ChatStreamRequest,
) (err error) {
	errnie.Debug("provider.buildMessages")

	if prvdr.params == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	messageList := make([]*cohere.Message, 0, len(prvdr.params.Messages))

	for _, message := range prvdr.params.Messages {
		switch message.Role {
		case "system":
			chatParams.Preamble = cohere.String(message.Content)
		case "user":
			messageList = append(messageList, &cohere.Message{
				Role: "user",
				User: &cohere.ChatMessage{
					Message: message.Content,
				},
			})
		case "assistant":
			messageList = append(messageList, &cohere.Message{
				Role: "chatbot",
				Chatbot: &cohere.ChatMessage{
					Message: message.Content,
				},
			})
		default:
			errnie.Error("unknown message role", "role", message.Role)
		}
	}

	chatParams.ChatHistory = messageList
	return nil
}

func (prvdr *CohereProvider) buildTools(
	chatParams *cohere.ChatStreamRequest,
) (err error) {
	errnie.Debug("provider.buildTools")

	if prvdr.params == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	toolList := make([]*cohere.Tool, 0, len(prvdr.params.Tools))

	for _, tool := range prvdr.params.Tools {
		parameterDefinitions := make(
			map[string]*cohere.ToolParameterDefinitionsValue,
			len(tool.Function.Parameters.Properties),
		)

		for _, property := range tool.Function.Parameters.Properties {
			parameterDefinitions[property.Name] = &cohere.ToolParameterDefinitionsValue{
				Type:        property.Type,
				Description: cohere.String(property.Description),
				Required:    cohere.Bool(true),
			}
		}

		toolList = append(toolList, &cohere.Tool{
			Name:                 tool.Function.Name,
			Description:          tool.Function.Description,
			ParameterDefinitions: parameterDefinitions,
		})
	}

	if len(toolList) > 0 {
		chatParams.Tools = toolList
	}

	return nil
}

func (prvdr *CohereProvider) buildResponseFormat(
	chatParams *cohere.ChatStreamRequest,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	if prvdr.params == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	var (
		schema map[string]interface{}
		buf    []byte
	)

	if buf, err = json.Marshal(&prvdr.params.ResponseFormat.Schema); err != nil {
		return errnie.Error(err)
	}

	if err = json.Unmarshal(buf, &schema); err != nil {
		return errnie.Error(err)
	}

	chatParams.ResponseFormat = &cohere.ResponseFormat{
		Type: "json_object",
		JsonObject: &cohere.JsonResponseFormat{
			Schema: schema,
		},
	}

	return nil
}

type CohereEmbedder struct {
	params   *Params
	apiKey   string
	endpoint string
	client   *cohereclient.Client
}

func NewCohereEmbedder(apiKey string, endpoint string) *CohereEmbedder {
	errnie.Debug("provider.NewCohereEmbedder")

	if apiKey == "" {
		apiKey = os.Getenv("COHERE_API_KEY")
	}

	return &CohereEmbedder{
		params:   &Params{},
		apiKey:   apiKey,
		endpoint: endpoint,
		client:   cohereclient.NewClient(cohereclient.WithToken(apiKey)),
	}
}

func (embedder *CohereEmbedder) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.CohereEmbedder.Read", "p", string(p))
	return 0, nil
}

func (embedder *CohereEmbedder) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.CohereEmbedder.Write")

	if len(embedder.params.Messages) == 0 {
		return len(p), nil
	}

	message := embedder.params.Messages[0]
	content := message.Content

	in := cohere.EmbedInputType(cohere.EmbedInputTypeSearchDocument)

	embedRequest := cohere.EmbedRequest{
		Texts:     []string{content},
		Model:     cohere.String("embed-english-v3.0"),
		InputType: &in,
	}

	response, err := embedder.client.Embed(context.Background(), &embedRequest)
	if err != nil {
		errnie.Error("embedding request failed", "error", err)
		return 0, err
	}

	if response != nil && len(response.EmbeddingsFloats.Embeddings) > 0 {
		embeddings := response.EmbeddingsFloats.Embeddings
		errnie.Debug("created embeddings",
			"text_length", len(content),
			"dimensions", len(embeddings),
		)
	}

	return len(p), nil
}

func (embedder *CohereEmbedder) Close() error {
	errnie.Debug("provider.CohereEmbedder.Close")
	embedder.params = nil
	return nil
}
