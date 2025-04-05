package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"

	"capnproto.org/go/capnp/v3"
	"github.com/mark3labs/mcp-go/mcp"
	aicontext "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/ai/params"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
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
	ctx      context.Context
	cancel   context.CancelFunc
	params   params.Params
	buffer   io.Writer
	segment  *capnp.Segment
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
		ctx:      ctx,
		cancel:   cancel,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *GoogleProvider) ID() string {
	return "google"
}

type GoogleProviderOption func(*GoogleProvider)

func WithGoogleAPIKey(apiKey string) GoogleProviderOption {
	return func(prvdr *GoogleProvider) {
		prvdr.client, _ = genai.NewClient(context.Background(), &genai.ClientConfig{
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
	params params.Params,
	ctx aicontext.Context,
	tools []mcp.Tool,
) chan *datura.ArtifactBuilder {
	errnie.Debug("provider.GoogleProvider.Generate")

	out := make(chan *datura.ArtifactBuilder)

	// Store the params for use in other methods
	prvdr.params = params

	go func() {
		defer close(out)

		chatConfig := &genai.GenerateContentConfig{
			Temperature:      utils.Ptr(float32(params.Temperature())),
			TopP:             utils.Ptr(float32(params.TopP())),
			TopK:             utils.Ptr(float32(params.TopK())),
			FrequencyPenalty: utils.Ptr(float32(params.FrequencyPenalty())),
			PresencePenalty:  utils.Ptr(float32(params.PresencePenalty())),
		}

		if params.MaxTokens() > 1 {
			chatConfig.MaxOutputTokens = utils.Ptr(int32(params.MaxTokens()))
		}

		messages, err := prvdr.buildMessages(chatConfig, ctx)
		if err != nil {
			out <- datura.New(datura.WithError(err))
			return
		}

		if err = prvdr.buildTools(chatConfig, tools); err != nil {
			out <- datura.New(datura.WithError(err))
			return
		}

		format, err := params.Format()

		if errnie.Error(err) != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		if err = prvdr.buildResponseFormat(chatConfig, format); err != nil {
			out <- datura.New(datura.WithError(err))
			return
		}

		model, err := params.Model()

		if err != nil {
			out <- datura.New(datura.WithError(err))
			return
		}

		if params.Stream() {
			if err = prvdr.handleStreamingRequest(chatConfig, messages); err != nil {
				out <- datura.New(datura.WithError(err))
				return
			}
		} else {
			if err = prvdr.handleSingleRequest(chatConfig, model, messages); err != nil {
				out <- datura.New(datura.WithError(err))
				return
			}
		}
	}()

	return out
}

func (prvdr *GoogleProvider) Name() string {
	return "google"
}

func (prvdr *GoogleProvider) handleSingleRequest(
	chatConfig *genai.GenerateContentConfig,
	model string,
	messages []*genai.Content,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	resp, err := prvdr.client.Models.GenerateContent(
		context.Background(),
		model,
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

	msg, err := aicontext.NewMessage(prvdr.segment)

	if errnie.Error(err) != nil {
		return errnie.Error(err)
	}

	if err = msg.SetRole("assistant"); errnie.Error(err) != nil {
		return errnie.Error(err)
	}

	if err = msg.SetContent(""); errnie.Error(err) != nil {
		return errnie.Error(err)
	}

	contentText := ""
	// Combine all parts into the content
	for _, part := range resp.Candidates[0].Content.Parts {
		contentText += part.Text
	}

	if err = msg.SetContent(contentText); errnie.Error(err) != nil {
		return errnie.Error(err)
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

		toolCalls, err := msg.NewToolCalls(1)
		if errnie.Error(err) != nil {
			return errnie.Error(err)
		}

		if err = toolCalls.At(0).SetId(fc.Name); errnie.Error(err) != nil {
			return errnie.Error(err)
		}

		if err = toolCalls.At(0).SetName(fc.Name); errnie.Error(err) != nil {
			return errnie.Error(err)
		}

		if err = toolCalls.At(0).SetArguments(argsStr); errnie.Error(err) != nil {
			return errnie.Error(err)
		}

		errnie.Info("toolCall detected", "name", fc.Name)
	}

	if _, err = io.Copy(prvdr.buffer, datura.New(
		datura.WithPayload([]byte(contentText)),
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

	model, err := prvdr.params.Model()
	if err != nil {
		return errnie.Error(err)
	}

	for response, err := range prvdr.client.Models.GenerateContentStream(
		prvdr.ctx,
		model,
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

				msg, err := aicontext.NewMessage(prvdr.segment)
				if errnie.Error(err) != nil {
					return errnie.Error(err)
				}

				if err = msg.SetRole("assistant"); errnie.Error(err) != nil {
					return errnie.Error(err)
				}

				if err = msg.SetContent(""); errnie.Error(err) != nil {
					return errnie.Error(err)
				}

				toolCalls, err := msg.NewToolCalls(1)
				if errnie.Error(err) != nil {
					return errnie.Error(err)
				}

				if err = toolCalls.At(0).SetId(fc.Name); errnie.Error(err) != nil {
					return errnie.Error(err)
				}

				if err = toolCalls.At(0).SetName(fc.Name); errnie.Error(err) != nil {
					return errnie.Error(err)
				}

				if err = toolCalls.At(0).SetArguments(argsStr); errnie.Error(err) != nil {
					return errnie.Error(err)
				}

				errnie.Info("toolCall detected (streaming)", "name", fc.Name)
			}
		}
	}

	return nil
}

func (prvdr *GoogleProvider) buildMessages(
	chatConfig *genai.GenerateContentConfig,
	ctx aicontext.Context,
) (messages []*genai.Content, err error) {
	errnie.Debug("provider.buildMessages")

	msgs, err := ctx.Messages()
	if err != nil {
		return nil, errnie.Error(err)
	}

	messages = make([]*genai.Content, 0, msgs.Len())

	for i := 0; i < msgs.Len(); i++ {
		msg := msgs.At(i)

		role, err := msg.Role()
		if err != nil {
			return nil, errnie.Error(err)
		}

		msgContent, err := msg.Content()
		if err != nil {
			return nil, errnie.Error(err)
		}

		msgParts := []*genai.Part{{Text: msgContent}}

		// Handle tool calls for assistant messages
		if role == "assistant" {
			toolCalls, err := msg.ToolCalls()
			if err != nil {
				return nil, errnie.Error(err)
			}

			for j := 0; j < toolCalls.Len(); j++ {
				toolCall := toolCalls.At(j)

				name, err := toolCall.Name()
				if err != nil {
					return nil, errnie.Error(err)
				}

				arguments, err := toolCall.Arguments()
				if err != nil {
					return nil, errnie.Error(err)
				}

				// Parse arguments string back to map
				var args map[string]interface{}
				if err := json.Unmarshal([]byte(arguments), &args); err != nil {
					errnie.Error("failed to parse tool call arguments", "error", err)
					continue
				}

				msgParts = append(msgParts, &genai.Part{
					FunctionCall: &genai.FunctionCall{
						Name: name,
						Args: args,
					},
				})
			}
		}

		content := &genai.Content{
			Role:  role,
			Parts: msgParts,
		}

		messages = append(messages, content)
	}

	return messages, nil
}

func (prvdr *GoogleProvider) buildTools(
	chatConfig *genai.GenerateContentConfig,
	tools []mcp.Tool,
) (err error) {
	errnie.Debug("provider.buildTools")

	if len(tools) == 0 {
		chatConfig.Tools = nil
		return nil
	}

	googleTools := make([]*genai.Tool, 0, len(tools))

	for _, tool := range tools {
		functionDeclaration := &genai.FunctionDeclaration{
			Name:        tool.Name,
			Description: tool.Description,
		}

		// Convert parameters to Google's format
		properties := tool.InputSchema.Properties
		required := tool.InputSchema.Required

		if len(properties) > 0 {
			schema := &genai.Schema{
				Type:       genai.Type("object"),
				Properties: make(map[string]*genai.Schema),
				Required:   required,
			}

			for propName, propValue := range properties {
				propMap, ok := propValue.(map[string]interface{})
				if !ok {
					continue
				}

				propType, _ := propMap["type"].(string)
				propDescription, _ := propMap["description"].(string)

				// TODO: Handle enum properly
				enumValues := []string{}

				schema.Properties[propName] = &genai.Schema{
					Type:        genai.Type(propType),
					Description: propDescription,
					Enum:        enumValues,
				}
			}

			functionDeclaration.Parameters = schema
		}

		googleTools = append(googleTools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{functionDeclaration},
		})
	}

	chatConfig.Tools = googleTools
	return nil
}

func (prvdr *GoogleProvider) buildResponseFormat(
	chatConfig *genai.GenerateContentConfig,
	format params.ResponseFormat,
) (err error) {
	errnie.Debug("provider.buildResponseFormat")

	// Check if format is a zero value or nil pointer
	name, err := format.Name()
	if err != nil || name == "" {
		return nil
	}

	description, err := format.Description()
	if err != nil {
		return errnie.Error(err)
	}

	// Google's API doesn't have direct JSON schema support like OpenAI
	// Instead, we'll add it to the system message
	systemMsg := fmt.Sprintf(
		"Please format your response as a JSON object following this schema:\n%s\n%s",
		name,
		description,
	)

	schema, err := format.Schema()
	if err != nil {
		return errnie.Error(err)
	}

	if schema != "" {
		systemMsg += fmt.Sprintf("\nSchema: %v", schema)
	}

	chatConfig.SystemInstruction = &genai.Content{
		Role:  "system",
		Parts: []*genai.Part{{Text: systemMsg}},
	}

	return nil
}

type GoogleEmbedder struct {
	apiKey   string
	endpoint string
	client   *genai.Client
}

func NewGoogleEmbedder(apiKey string, endpoint string) *GoogleEmbedder {
	errnie.Debug("provider.NewGoogleEmbedder")

	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})

	if errnie.Error(err) != nil {
		return nil
	}

	return &GoogleEmbedder{
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
	return nil
}
