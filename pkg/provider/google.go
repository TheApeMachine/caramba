package provider

import (
	"context"
	"io"
	"os"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/utils"
	"google.golang.org/genai"
)

/*
GoogleProvider implements an LLM provider that connects to Google's Gemini API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type GoogleProvider struct {
	client *genai.Client
	buffer *stream.Buffer
	params *Params
	ctx    context.Context
	cancel context.CancelFunc
}

/*
NewGoogleProvider creates a new Google Gemini provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the GOOGLE_API_KEY environment variable.
*/
func NewGoogleProvider(
	apiKey string,
	endpoint string,
) *GoogleProvider {
	errnie.Debug("provider.NewGoogleProvider")

	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}

	if endpoint == "" {
		endpoint = viper.GetViper().GetString("endpoints.google")
	}

	ctx, cancel := context.WithCancel(context.Background())
	params := &Params{}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})

	if err != nil {
		errnie.Error("failed to create Google client", "error", err)
		return nil
	}

	return &GoogleProvider{
		client: client,
		buffer: stream.NewBuffer(func(artfct *datura.Artifact) (err error) {
			var payload []byte

			if payload, err = artfct.EncryptedPayload(); err != nil {
				return errnie.Error(err)
			}

			params.Unmarshal(payload)
			return nil
		}),
		params: params,
		ctx:    ctx,
		cancel: cancel,
	}
}

/*
Read implements the io.Reader interface.
*/
func (prvdr *GoogleProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.GoogleProvider.Read")
	return prvdr.buffer.Read(p)
}

/*
Write implements the io.Writer interface.
*/
func (prvdr *GoogleProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.GoogleProvider.Write")

	n, err = prvdr.buffer.Write(p)
	if errnie.Error(err) != nil {
		return n, err
	}

	chatConfig := &genai.GenerateContentConfig{
		Temperature:      utils.Ptr[float32](float32(prvdr.params.Temperature)),
		TopP:             utils.Ptr[float32](float32(prvdr.params.TopP)),
		TopK:             utils.Ptr[float32](float32(prvdr.params.TopK)),
		FrequencyPenalty: utils.Ptr[float32](float32(prvdr.params.FrequencyPenalty)),
		PresencePenalty:  utils.Ptr[float32](float32(prvdr.params.PresencePenalty)),
		MaxOutputTokens:  utils.Ptr[int32](int32(prvdr.params.MaxTokens)),
	}

	messages := prvdr.buildMessages(chatConfig)
	prvdr.buildTools(chatConfig)
	prvdr.buildResponseFormat(chatConfig)

	if prvdr.params.Stream {
		prvdr.handleStreamingRequest(messages, chatConfig)
	} else {
		prvdr.handleSingleRequest(messages, chatConfig)
	}

	return n, nil
}

/*
Close cleans up any resources.
*/
func (prvdr *GoogleProvider) Close() error {
	errnie.Debug("provider.GoogleProvider.Close")
	prvdr.cancel()
	return nil
}

func (prvdr *GoogleProvider) handleSingleRequest(
	messages []*genai.Content,
	chatConfig *genai.GenerateContentConfig,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	resp, err := prvdr.client.Models.GenerateContent(
		prvdr.ctx,
		prvdr.params.Model,
		messages,
		chatConfig,
	)

	if errnie.Error(err) != nil {
		return
	}

	if len(resp.Candidates) == 0 {
		errnie.Error("no response candidates")
		return
	}

	content := ""
	for _, part := range resp.Candidates[0].Content.Parts {
		content += string(part.Text)
	}

	if _, err = io.Copy(prvdr, datura.New(
		datura.WithPayload([]byte(content)),
	)); errnie.Error(err) != nil {
		return err
	}

	return nil
}

func (prvdr *GoogleProvider) handleStreamingRequest(
	messages []*genai.Content,
	chatConfig *genai.GenerateContentConfig,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	for response, err := range prvdr.client.Models.GenerateContentStream(
		prvdr.ctx,
		prvdr.params.Model,
		messages,
		chatConfig,
	) {
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			errnie.Error("streaming error", "error", err)
			return err
		}

		if len(response.Candidates) == 0 {
			continue
		}

		text := ""

		if text, err = response.Text(); errnie.Error(err) != nil {
			continue
		}

		if _, err = io.Copy(prvdr, datura.New(
			datura.WithPayload([]byte(text)),
		)); errnie.Error(err) != nil {
			continue
		}
	}

	return nil
}

func (prvdr *GoogleProvider) buildMessages(
	chatParams *genai.GenerateContentConfig,
) []*genai.Content {
	errnie.Debug("provider.buildMessages")

	if prvdr.params == nil {
		errnie.NewErrValidation("params are nil", "provider", "google")
		return nil
	}

	messages := make([]*genai.Content, 0, len(prvdr.params.Messages))

	for _, message := range prvdr.params.Messages {
		switch message.Role {
		case "system":
			chatParams.SystemInstruction = &genai.Content{
				Role:  "system",
				Parts: []*genai.Part{{Text: message.Content}},
			}
		case "user":
			messages = append(messages, &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: message.Content}},
			})
		case "assistant":
			messages = append(messages, &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{{Text: message.Content}},
			})
		default:
			errnie.Error("unknown message role", "role", message.Role)
		}
	}

	return messages
}

func (prvdr *GoogleProvider) buildTools(
	chatParams *genai.GenerateContentConfig,
) {
	errnie.Debug("provider.buildTools")

	if prvdr.params == nil {
		errnie.NewErrValidation("params are nil", "provider", "google")
		return
	}

	if len(prvdr.params.Tools) == 0 {
		chatParams.Tools = nil
		return
	}

	functionDeclarations := make([]*genai.FunctionDeclaration, 0, len(prvdr.params.Tools))

	for _, tool := range prvdr.params.Tools {
		functionDeclarations = append(functionDeclarations, &genai.FunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
		})
	}

	chatParams.Tools = []*genai.Tool{
		{
			FunctionDeclarations: functionDeclarations,
		},
	}
}

func (prvdr *GoogleProvider) buildResponseFormat(
	chatParams *genai.GenerateContentConfig,
) {
	errnie.Debug("provider.buildResponseFormat")

	if prvdr.params == nil {
		errnie.NewErrValidation("params are nil", "provider", "google")
		return
	}

	// Add format instructions to system message
	if prvdr.params.ResponseFormat.Name != "" {
		chatParams.ResponseSchema = &genai.Schema{
			Title:       prvdr.params.ResponseFormat.Name,
			Description: prvdr.params.ResponseFormat.Description,
		}
	}
}

type GoogleEmbedder struct {
	params   *Params
	apiKey   string
	endpoint string
	client   *genai.Client
}

func NewGoogleEmbedder(apiKey string, endpoint string) *GoogleEmbedder {
	errnie.Debug("provider.NewGoogleEmbedder")

	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		errnie.Error("failed to create Google embedder client", "error", err)
		return nil
	}

	return &GoogleEmbedder{
		params:   &Params{},
		apiKey:   apiKey,
		endpoint: endpoint,
		client:   client,
	}
}

func (embedder *GoogleEmbedder) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.GoogleEmbedder.Read", "p", string(p))
	return 0, nil
}

func (embedder *GoogleEmbedder) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.GoogleEmbedder.Write")
	return len(p), nil
}

func (embedder *GoogleEmbedder) Close() error {
	errnie.Debug("provider.GoogleEmbedder.Close")
	embedder.params = nil
	return nil
}
