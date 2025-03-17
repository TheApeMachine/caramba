package provider

import (
	"context"
	"encoding/json"
	"os"

	"github.com/google/generative-ai-go/genai"
	"github.com/spf13/viper"
	aiCtx "github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/event"
	"github.com/theapemachine/caramba/pkg/message"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/utils"
	"google.golang.org/api/option"
)

/*
GoogleProvider implements an LLM provider that connects to Google's Gemini API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type GoogleProvider struct {
	client *genai.Client
	model  *genai.GenerativeModel
	buffer *stream.Buffer
	params *aiCtx.Artifact
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

	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		errnie.Error("failed to create Google client", "error", err)
		return nil
	}

	prvdr := &GoogleProvider{
		client: client,
		model:  client.GenerativeModel("gemini-1.5-pro"),
		params: aiCtx.New(
			"gemini-1.5-pro",
			nil,
			nil,
			nil,
			0.7,
			1.0,
			0,
			0.0,
			0.0,
			2048,
			false,
		),
		ctx:    ctx,
		cancel: cancel,
	}

	prvdr.buffer = stream.NewBuffer(
		func(event *event.Artifact) error {
			errnie.Debug("provider.GoogleProvider.buffer.fn", "event", event)

			payload, err := event.Payload()
			if errnie.Error(err) != nil {
				return err
			}

			_, err = prvdr.params.Write(payload)
			if errnie.Error(err) != nil {
				return err
			}

			return nil
		},
	)

	return prvdr
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

	chat := prvdr.model.StartChat()

	prvdr.buildMessages(prvdr.params, chat)
	prvdr.buildTools(prvdr.params)
	prvdr.buildResponseFormat(prvdr.params, chat)

	if prvdr.params.Stream() {
		prvdr.handleStreamingRequest(chat)
	} else {
		prvdr.handleSingleRequest(chat)
	}

	return n, nil
}

/*
Close cleans up any resources.
*/
func (prvdr *GoogleProvider) Close() error {
	errnie.Debug("provider.GoogleProvider.Close")
	prvdr.cancel()
	if prvdr.client != nil {
		prvdr.client.Close()
	}
	return prvdr.params.Close()
}

func (prvdr *GoogleProvider) buildMessages(
	params *aiCtx.Artifact,
	chat *genai.ChatSession,
) {
	errnie.Debug("provider.buildMessages")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "google")
		return
	}

	messages, err := params.Messages()
	if err != nil {
		errnie.Error("failed to get messages", "error", err)
		return
	}

	for idx := range messages.Len() {
		message := messages.At(idx)

		role, err := message.Role()
		if err != nil {
			errnie.Error("failed to get message role", "error", err)
			continue
		}

		content, err := message.Content()
		if err != nil {
			errnie.Error("failed to get message content", "error", err)
			continue
		}

		msgContent := &genai.Content{
			Parts: []genai.Part{genai.Text(content)},
		}

		switch role {
		case "system":
			msgContent.Role = "user"
			chat.History = append(chat.History, msgContent)
		case "user":
			msgContent.Role = "user"
			chat.History = append(chat.History, msgContent)
		case "assistant":
			msgContent.Role = "model"
			chat.History = append(chat.History, msgContent)
		default:
			errnie.Error("unknown message role", "role", role)
		}
	}
}

func (prvdr *GoogleProvider) buildTools(
	params *aiCtx.Artifact,
) {
	errnie.Debug("provider.buildTools")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "google")
		return
	}

	tools, err := params.Tools()
	if err != nil {
		errnie.Error("failed to get tools", "error", err)
		return
	}

	if tools.Len() == 0 {
		prvdr.model.Tools = nil
		return
	}

	functionDeclarations := make([]*genai.FunctionDeclaration, 0, tools.Len())

	for idx := range tools.Len() {
		tool := tools.At(idx)

		name, err := tool.Name()
		if err != nil {
			errnie.Error("failed to get tool name", "error", err)
			continue
		}

		description, err := tool.Description()
		if err != nil {
			errnie.Error("failed to get tool description", "error", err)
			continue
		}

		functionDeclarations = append(functionDeclarations, &genai.FunctionDeclaration{
			Name:        name,
			Description: description,
		})
	}

	prvdr.model.Tools = []*genai.Tool{
		{
			FunctionDeclarations: functionDeclarations,
		},
	}
}

func (prvdr *GoogleProvider) buildResponseFormat(
	params *aiCtx.Artifact,
	chat *genai.ChatSession,
) {
	errnie.Debug("provider.buildResponseFormat")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "google")
		return
	}

	data, err := params.Process()
	if err != nil || data == nil {
		return
	}

	var formatData map[string]interface{}
	if err := json.Unmarshal(data, &formatData); err != nil {
		errnie.Error("failed to unmarshal process data", "error", err)
		return
	}

	// Add format instructions to system message
	if name, ok := formatData["name"].(string); ok {
		if desc, ok := formatData["description"].(string); ok {
			formatInstructions := &genai.Content{
				Role: "user",
				Parts: []genai.Part{genai.Text(
					"Please format your response according to the specified schema: " +
						name + ". " + desc,
				)},
			}
			chat.History = append(chat.History, formatInstructions)
		}
	}
}

func (prvdr *GoogleProvider) handleSingleRequest(
	chat *genai.ChatSession,
) (err error) {
	errnie.Debug("provider.handleSingleRequest")

	resp, err := chat.SendMessage(prvdr.ctx, genai.Text(""))
	if errnie.Error(err) != nil {
		return
	}

	if len(resp.Candidates) == 0 {
		errnie.Error("no response candidates")
		return
	}

	content := ""
	for _, part := range resp.Candidates[0].Content.Parts {
		if text, ok := part.(genai.Text); ok {
			content += string(text)
		}
	}

	return utils.SendEvent(
		prvdr.buffer,
		"provider.google",
		message.AssistantRole,
		content,
	)
}

func (prvdr *GoogleProvider) handleStreamingRequest(
	chat *genai.ChatSession,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	iter := chat.SendMessageStream(prvdr.ctx, genai.Text(""))

	for {
		resp, err := iter.Next()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			errnie.Error("streaming error", "error", err)
			return err
		}

		if len(resp.Candidates) == 0 {
			continue
		}

		for _, part := range resp.Candidates[0].Content.Parts {
			if text, ok := part.(genai.Text); ok {
				if err = utils.SendEvent(
					prvdr.buffer,
					"provider.google",
					message.AssistantRole,
					string(text),
				); errnie.Error(err) != nil {
					continue
				}
			}
		}
	}

	return nil
}

type GoogleEmbedder struct {
	params   *aiCtx.Artifact
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
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		errnie.Error("failed to create Google embedder client", "error", err)
		return nil
	}

	return &GoogleEmbedder{
		params:   &aiCtx.Artifact{},
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

	messages, err := embedder.params.Messages()
	if err != nil {
		errnie.Error("failed to get messages", "error", err)
		return 0, err
	}

	if messages.Len() == 0 {
		return len(p), nil
	}

	message := messages.At(0)
	content, err := message.Content()
	if err != nil {
		errnie.Error("failed to get message content", "error", err)
		return 0, err
	}

	embeddingModel := embedder.client.EmbeddingModel("text-embedding-004")
	result, err := embeddingModel.EmbedContent(context.Background(), genai.Text(content))
	if err != nil {
		errnie.Error("embedding request failed", "error", err)
		return 0, err
	}

	if result != nil && result.Embedding != nil {
		errnie.Debug("created embeddings",
			"text_length", len(content),
			"dimensions", len(result.Embedding.Values),
		)
	}

	return len(p), nil
}

func (embedder *GoogleEmbedder) Close() error {
	errnie.Debug("provider.GoogleEmbedder.Close")
	embedder.params = nil
	if embedder.client != nil {
		embedder.client.Close()
	}
	return nil
}
