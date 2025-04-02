package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"

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
	client   *genai.Client
	endpoint string
	buffer   *stream.Buffer
	params   *Params
	ctx      context.Context
	cancel   context.CancelFunc
}

/*
NewGoogleProvider creates a new Google Gemini provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the GOOGLE_API_KEY environment variable.
*/
func NewGoogleProvider(opts ...GoogleProviderOption) *GoogleProvider {
	errnie.Debug("provider.NewGoogleProvider")

	ctx, cancel := context.WithCancel(context.Background())

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  os.Getenv("GOOGLE_API_KEY"),
		Backend: genai.BackendGeminiAPI,
	})

	if err != nil {
		errnie.Error("failed to create Google client", "error", err)
		cancel()
		return nil
	}

	prvdr := &GoogleProvider{
		client:   client,
		endpoint: "",
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

type GoogleProviderOption func(*GoogleProvider)

func WithGoogleAPIKey(apiKey string) GoogleProviderOption {
	return func(prvdr *GoogleProvider) {
		prvdr.client, _ = genai.NewClient(prvdr.ctx, &genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		})
	}
}

func WithGoogleEndpoint(endpoint string) GoogleProviderOption {
	return func(prvdr *GoogleProvider) {
		prvdr.endpoint = endpoint
	}
}

func (prvdr *GoogleProvider) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("provider.GoogleProvider.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		select {
		case <-prvdr.ctx.Done():
			errnie.Debug("provider.GoogleProvider.Generate.ctx.Done")
			prvdr.cancel()
			return
		case artifact := <-buffer:
			if err := artifact.To(prvdr.params); err != nil {
				out <- datura.New(datura.WithError(errnie.Error(err)))
				return
			}

			chatConfig := &genai.GenerateContentConfig{
				Temperature:      utils.Ptr(float32(prvdr.params.Temperature)),
				TopP:             utils.Ptr(float32(prvdr.params.TopP)),
				TopK:             utils.Ptr(float32(prvdr.params.TopK)),
				FrequencyPenalty: utils.Ptr(float32(prvdr.params.FrequencyPenalty)),
				PresencePenalty:  utils.Ptr(float32(prvdr.params.PresencePenalty)),
			}

			if prvdr.params.MaxTokens > 1 {
				chatConfig.MaxOutputTokens = utils.Ptr(int32(prvdr.params.MaxTokens))
			}

			messages, err := prvdr.buildMessages()
			if err != nil {
				out <- datura.New(datura.WithError(err))
				return
			}

			if err = prvdr.buildTools(chatConfig); err != nil {
				out <- datura.New(datura.WithError(err))
				return
			}

			if prvdr.params.ResponseFormat != nil {
				if err = prvdr.buildResponseFormat(chatConfig); err != nil {
					out <- datura.New(datura.WithError(err))
					return
				}
			}

			if prvdr.params.Stream {
				if err = prvdr.handleStreamingRequest(chatConfig, messages); err != nil {
					out <- datura.New(datura.WithError(err))
					return
				}
			} else {
				if err = prvdr.handleSingleRequest(chatConfig, messages); err != nil {
					out <- datura.New(datura.WithError(err))
					return
				}
			}

			out <- datura.New(datura.WithPayload(prvdr.params.Marshal()))
		}
	}()

	return out
}

func (prvdr *GoogleProvider) Name() string {
	return "google"
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
		Temperature:      utils.Ptr(float32(prvdr.params.Temperature)),
		TopP:             utils.Ptr(float32(prvdr.params.TopP)),
		TopK:             utils.Ptr(float32(prvdr.params.TopK)),
		FrequencyPenalty: utils.Ptr(float32(prvdr.params.FrequencyPenalty)),
		PresencePenalty:  utils.Ptr(float32(prvdr.params.PresencePenalty)),
		MaxOutputTokens:  utils.Ptr(int32(prvdr.params.MaxTokens)),
	}

	messages, err := prvdr.buildMessages()
	if err != nil {
		return n, err
	}

	if err = prvdr.buildTools(chatConfig); err != nil {
		return n, errnie.Error(err)
	}

	if err = prvdr.buildResponseFormat(chatConfig); err != nil {
		return n, errnie.Error(err)
	}

	if prvdr.params.Stream {
		err = prvdr.handleStreamingRequest(chatConfig, messages)
	} else {
		err = prvdr.handleSingleRequest(chatConfig, messages)
	}

	return n, errnie.Error(err)
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
	chatConfig *genai.GenerateContentConfig,
	messages []*genai.Content,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	resp, err := prvdr.client.Models.GenerateContent(
		prvdr.ctx,
		prvdr.params.Model,
		messages,
		chatConfig,
	)

	if err != nil {
		return errnie.Error(err)
	}

	if len(resp.Candidates) == 0 {
		err = errors.New("no response candidates")
		return errnie.Error(err)
	}

	msg := &Message{
		Role:    MessageRoleAssistant,
		Name:    prvdr.params.Model,
		Content: "",
	}

	// Combine all parts into the content
	for _, part := range resp.Candidates[0].Content.Parts {
		msg.Content += part.Text
	}

	// Handle tool calls if present
	if resp.Candidates[0].Content.Parts[0].FunctionCall != nil {
		fc := resp.Candidates[0].Content.Parts[0].FunctionCall
		argsStr := ""
		for k, v := range fc.Args {
			argsStr += fmt.Sprintf(`"%s":%v,`, k, v)
		}
		if len(argsStr) > 0 {
			argsStr = "{" + argsStr[:len(argsStr)-1] + "}"
		}

		msg.ToolCalls = []ToolCall{
			{
				ID:   fc.Name, // Using name as ID since Google API doesn't provide separate ID
				Type: "function",
				Function: ToolCallFunction{
					Name:      fc.Name,
					Arguments: argsStr,
				},
			},
		}
		errnie.Info("toolCall detected", "name", fc.Name)
	}

	prvdr.params.Messages = append(prvdr.params.Messages, msg)

	if _, err = io.Copy(prvdr.buffer, datura.New(
		datura.WithPayload(prvdr.params.Marshal()),
	)); err != nil {
		return errnie.Error(err)
	}

	return nil
}

func (prvdr *GoogleProvider) handleStreamingRequest(
	chatConfig *genai.GenerateContentConfig,
	messages []*genai.Content,
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
			continue
		}

		if len(response.Candidates) == 0 {
			continue
		}

		for _, part := range response.Candidates[0].Content.Parts {
			if part.Text != "" {
				if _, err = io.Copy(prvdr.buffer, datura.New(
					datura.WithPayload([]byte(part.Text)),
				)); err != nil {
					errnie.Error("failed to write stream chunk", "error", err)
					continue
				}
			}

			// Handle tool calls in streaming mode
			if part.FunctionCall != nil {
				fc := part.FunctionCall
				argsStr := ""
				for k, v := range fc.Args {
					argsStr += fmt.Sprintf(`"%s":%v,`, k, v)
				}
				if len(argsStr) > 0 {
					argsStr = "{" + argsStr[:len(argsStr)-1] + "}"
				}

				msg := &Message{
					Role:    MessageRoleAssistant,
					Name:    prvdr.params.Model,
					Content: "",
					ToolCalls: []ToolCall{
						{
							ID:   fc.Name,
							Type: "function",
							Function: ToolCallFunction{
								Name:      fc.Name,
								Arguments: argsStr,
							},
						},
					},
				}

				prvdr.params.Messages = append(prvdr.params.Messages, msg)
				errnie.Info("toolCall detected (streaming)", "name", fc.Name)
			}
		}
	}

	return nil
}

func (prvdr *GoogleProvider) buildMessages() (messages []*genai.Content, err error) {
	errnie.Debug("provider.buildMessages")

	if prvdr.params == nil {
		return nil, errnie.BadRequest(errors.New("params are nil"))
	}

	messages = make([]*genai.Content, 0, len(prvdr.params.Messages))

	for _, message := range prvdr.params.Messages {
		content := &genai.Content{
			Role:  string(message.Role),
			Parts: []*genai.Part{{Text: message.Content}},
		}

		// Handle tool calls for assistant messages
		if message.Role == "assistant" && len(message.ToolCalls) > 0 {
			for _, toolCall := range message.ToolCalls {
				// Parse arguments string back to map
				var args map[string]interface{}
				if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
					errnie.Error("failed to parse tool call arguments", "error", err)
					continue
				}

				content.Parts = append(content.Parts, &genai.Part{
					FunctionCall: &genai.FunctionCall{
						Name: toolCall.Function.Name,
						Args: args,
					},
				})
			}
		}

		messages = append(messages, content)
	}

	return messages, nil
}

func (prvdr *GoogleProvider) buildTools(
	chatConfig *genai.GenerateContentConfig,
) (err error) {
	errnie.Debug("provider.buildTools")

	if prvdr.params == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	if len(prvdr.params.Tools) == 0 {
		chatConfig.Tools = nil
		return nil
	}

	tools := make([]*genai.Tool, 0, len(prvdr.params.Tools))

	for _, tool := range prvdr.params.Tools {
		functionDeclaration := &genai.FunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
		}

		// Convert parameters to Google's format
		if tool.Function.Parameters.Properties != nil {
			schema := &genai.Schema{
				Type:       genai.Type("object"),
				Properties: make(map[string]*genai.Schema),
				Required:   tool.Function.Parameters.Required,
			}

			for _, prop := range tool.Function.Parameters.Properties {
				enumStr := make([]string, 0)
				for _, e := range prop.Enum {
					if s, ok := e.(string); ok {
						enumStr = append(enumStr, s)
					}
				}

				schema.Properties[prop.Name] = &genai.Schema{
					Type:        genai.Type(prop.Type),
					Description: prop.Description,
					Enum:        enumStr,
				}
			}

			functionDeclaration.Parameters = schema
		}

		tools = append(tools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{functionDeclaration},
		})
	}

	chatConfig.Tools = tools
	return nil
}

func (prvdr *GoogleProvider) buildResponseFormat(
	chatConfig *genai.GenerateContentConfig,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	if prvdr.params == nil {
		return errnie.BadRequest(errors.New("params are nil"))
	}

	if prvdr.params.ResponseFormat == nil {
		return nil
	}

	// Google's API doesn't have direct JSON schema support like OpenAI
	// Instead, we'll add it to the system message
	systemMsg := fmt.Sprintf(
		"Please format your response as a JSON object following this schema:\n%s\n%s",
		prvdr.params.ResponseFormat.Name,
		prvdr.params.ResponseFormat.Description,
	)

	if prvdr.params.ResponseFormat.Schema != nil {
		systemMsg += fmt.Sprintf("\nSchema: %v", prvdr.params.ResponseFormat.Schema)
	}

	chatConfig.SystemInstruction = &genai.Content{
		Role:  "system",
		Parts: []*genai.Part{{Text: systemMsg}},
	}

	return nil
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

	if errnie.Error(err) != nil {
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
