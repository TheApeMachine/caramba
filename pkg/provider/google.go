package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"

	"github.com/google/generative-ai-go/genai"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
	"google.golang.org/api/option"
)

/*
GoogleProvider implements an LLM provider that connects to Google's Gemini API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type GoogleProvider struct {
	*ProviderData
	client *genai.Client
	model  *genai.GenerativeModel
	buffer *bufio.ReadWriter
	enc    *json.Encoder
	dec    *json.Decoder
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

	buf := bytes.NewBuffer([]byte{})
	buffer := bufio.NewReadWriter(
		bufio.NewReader(buf),
		bufio.NewWriter(buf),
	)

	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		errnie.Error("failed to create Google client", "error", err)
		return nil
	}

	model := client.GenerativeModel("gemini-1.5-pro")

	p := &GoogleProvider{
		ProviderData: &ProviderData{
			Params: &ai.ContextData{},
			Result: &core.EventData{},
		},
		client: client,
		model:  model,
		buffer: buffer,
		enc:    json.NewEncoder(buffer),
		dec:    json.NewDecoder(buffer),
	}

	return p
}

/*
Read implements the io.Reader interface.
*/
func (provider *GoogleProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.GoogleProvider.Read")

	if err = provider.buffer.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if n, err = provider.buffer.Read(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	errnie.Debug("provider.GoogleProvider.Read", "n", n, "err", err)
	return n, err
}

/*
Write implements the io.Writer interface.
*/
func (provider *GoogleProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.GoogleProvider.Write", "p", string(p))

	if n, err = provider.buffer.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = json.Unmarshal(p, provider.ProviderData.Params); err != nil {
		errnie.NewErrIO(err)
		return 0, err
	}

	errnie.Debug("provider.GoogleProvider.Write", "n", n, "err", err)

	// Prepare the conversation
	chat := provider.model.StartChat()

	// Add system prompt if present
	systemPrompt := provider.findSystemPrompt(provider.ProviderData.Params)
	if systemPrompt != "" {
		chat.History = append(chat.History, &genai.Content{
			Role:  "system",
			Parts: []genai.Part{genai.Text(systemPrompt)},
		})
	}

	// Add messages to history
	provider.buildMessages(provider.ProviderData.Params, chat)

	// Configure function declarations
	provider.buildTools(provider.ProviderData.Params)

	// Generate response streaming
	err = errnie.NewErrIO(provider.handleStreamingRequest(chat))

	return n, err
}

/*
Close cleans up any resources.
*/
func (provider *GoogleProvider) Close() error {
	errnie.Debug("provider.GoogleProvider.Close")

	// Reset state
	provider.ProviderData.Params = nil
	provider.ProviderData.Result = nil

	provider.buffer = nil
	provider.enc = nil
	provider.dec = nil

	if provider.client != nil {
		provider.client.Close()
	}

	return nil
}

func (p *GoogleProvider) findSystemPrompt(
	params *ai.ContextData,
) string {
	errnie.Debug("provider.findSystemPrompt")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "google")
		return ""
	}

	// Find system prompt from messages
	var systemPrompt string
	for _, message := range params.Messages {
		if message.Role == "system" {
			systemPrompt = message.Content
			break
		}
	}

	// Add structured output instructions if needed
	if params.Process != nil && params.Process.ProcessData != nil && params.Process.ProcessData.Schema != nil {
		formatInstructions := "Please format your response according to the specified schema: " +
			params.Process.ProcessData.Name + ". " + params.Process.ProcessData.Description

		if systemPrompt != "" {
			systemPrompt = systemPrompt + "\n\n" + formatInstructions
		} else {
			systemPrompt = formatInstructions
		}
	}

	return systemPrompt
}

func (p *GoogleProvider) buildMessages(
	params *ai.ContextData,
	chat *genai.ChatSession,
) {
	errnie.Debug("provider.buildMessages")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "google")
		return
	}

	for _, message := range params.Messages {
		// Skip system messages as they are handled separately
		if message.Role == "system" {
			continue
		}

		content := &genai.Content{
			Parts: []genai.Part{genai.Text(message.Content)},
		}

		switch message.Role {
		case "user":
			content.Role = "user"
		case "assistant":
			content.Role = "model"
		default:
			errnie.Error("unknown message role", "role", message.Role)
			continue
		}

		chat.History = append(chat.History, content)
	}
}

func (provider *GoogleProvider) buildTools(params *ai.ContextData) {
	// If no tools are provided, don't set any tools
	if len(params.Tools) == 0 {
		provider.model.Tools = nil
		return
	}

	errnie.Debug("preparing tools for Google provider", "count", len(params.Tools))

	// Create function declarations for each tool
	functionDeclarations := make([]*genai.FunctionDeclaration, 0, len(params.Tools))

	for _, tool := range params.Tools {
		// Create a basic function declaration for each tool
		functionDeclaration := &genai.FunctionDeclaration{
			Name:        tool.Name,
			Description: tool.Description,
		}

		// We would ideally convert the schema, but let's keep it simple for now
		// as the schema conversion is complex and requires specific type handling

		functionDeclarations = append(functionDeclarations, functionDeclaration)
		errnie.Debug("added tool for Google provider", "name", tool.Name)
	}

	// Set the tools on the model
	provider.model.Tools = []*genai.Tool{
		{
			FunctionDeclarations: functionDeclarations,
		},
	}
}

/*
handleStreamingRequest processes a streaming completion request
and emits chunks as they're received.
*/
func (provider *GoogleProvider) handleStreamingRequest(
	chat *genai.ChatSession,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	ctx := context.Background()

	// Send the prompt to generate a response
	iter := chat.SendMessageStream(ctx, genai.Text(""))

	count := 0
	var contentBuffer bytes.Buffer

	for {
		resp, err := iter.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			errnie.Error("streaming error", "error", err)
			return errnie.NewErrHTTP(err, 500)
		}

		// Extract text content from response
		for _, part := range resp.Candidates[0].Content.Parts {
			if textPart, ok := part.(genai.Text); ok {
				content := string(textPart)
				contentBuffer.WriteString(content)

				errnie.Debug("received stream chunk", "content", content)

				provider.Result = core.NewEvent(
					core.NewMessage(
						"assistant",
						"google",
						content,
					),
					nil,
				).EventData

				errnie.Debug("provider.handleStreamingRequest", "result", provider.Result)

				if err = provider.enc.Encode(provider.Result); err != nil {
					errnie.NewErrIO(err)
					return err
				}

				count++
			}
		}
	}

	errnie.Debug("streaming completed", "chunks", count)
	return nil
}

type GoogleEmbedderData struct {
	Params *ai.ContextData `json:"params"`
	Result *[]float64      `json:"result"`
}

type GoogleEmbedder struct {
	*GoogleEmbedderData
	apiKey   string
	endpoint string
	client   *genai.Client
	enc      *json.Encoder
	dec      *json.Decoder
	in       *bufio.ReadWriter
	out      *bufio.ReadWriter
}

func NewGoogleEmbedder(apiKey string, endpoint string) *GoogleEmbedder {
	errnie.Debug("provider.NewGoogleEmbedder")

	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}

	in := bufio.NewReadWriter(
		bufio.NewReader(bytes.NewBuffer([]byte{})),
		bufio.NewWriter(bytes.NewBuffer([]byte{})),
	)
	out := bufio.NewReadWriter(
		bufio.NewReader(bytes.NewBuffer([]byte{})),
		bufio.NewWriter(bytes.NewBuffer([]byte{})),
	)

	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		errnie.Error("failed to create Google embedder client", "error", err)
		return nil
	}

	embedder := &GoogleEmbedder{
		GoogleEmbedderData: &GoogleEmbedderData{},
		apiKey:             apiKey,
		endpoint:           endpoint,
		client:             client,
		enc:                json.NewEncoder(out),
		dec:                json.NewDecoder(in),
		in:                 in,
		out:                out,
	}

	embedder.enc.Encode(embedder.GoogleEmbedderData)

	return embedder
}

func (embedder *GoogleEmbedder) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.GoogleEmbedder.Read", "p", string(p))

	if err = embedder.out.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	n, err = embedder.out.Read(p)

	if err != nil {
		errnie.NewErrIO(err)
	}

	return n, err
}

func (embedder *GoogleEmbedder) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.GoogleEmbedder.Write")

	if n, err = embedder.in.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = embedder.in.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = embedder.dec.Decode(embedder.GoogleEmbedderData); err != nil {
		errnie.NewErrIO(err)
		return
	}

	// Use the actual embedding API
	ctx := context.Background()

	// Get the text to embed
	var textToEmbed string
	if len(embedder.GoogleEmbedderData.Params.Messages) > 0 {
		textToEmbed = embedder.GoogleEmbedderData.Params.Messages[0].Content
	}

	// Create embedding model
	embeddingModel := embedder.client.EmbeddingModel("text-embedding-004")

	// Request embeddings using the proper method as per documentation
	embeddingResult, err := embeddingModel.EmbedContent(ctx, genai.Text(textToEmbed))

	if err != nil {
		errnie.Error("embedding request failed", "error", err)
		*embedder.GoogleEmbedderData.Result = make([]float64, 0)
	} else if embeddingResult != nil && embeddingResult.Embedding != nil {
		// The documentation shows embedding values should be available in Embedding.Values
		values := embeddingResult.Embedding.Values

		// Convert from []float32 to []float64
		float64Values := make([]float64, len(values))
		for i, v := range values {
			float64Values[i] = float64(v)
		}

		// Store the converted values
		*embedder.GoogleEmbedderData.Result = float64Values

		errnie.Debug("created embeddings",
			"text_length", len(textToEmbed),
			"dimensions", len(float64Values),
		)
	} else {
		errnie.Error("embedding response was empty or invalid")
		*embedder.GoogleEmbedderData.Result = make([]float64, 0)
	}

	if err = embedder.enc.Encode(embedder.GoogleEmbedderData); err != nil {
		errnie.NewErrIO(err)
		return
	}

	return len(p), nil
}

func (embedder *GoogleEmbedder) Close() error {
	errnie.Debug("provider.GoogleEmbedder.Close")

	embedder.GoogleEmbedderData.Params = nil
	embedder.GoogleEmbedderData.Result = nil

	if embedder.client != nil {
		embedder.client.Close()
	}

	return nil
}
