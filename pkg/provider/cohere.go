package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"time"

	cohere "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
CohereProvider implements an LLM provider that connects to Cohere's API.
It supports regular chat completions, tool calling, and structured outputs.
*/
type CohereProvider struct {
	*ProviderData
	client *cohereclient.Client
	buffer *bufio.ReadWriter
	enc    *json.Encoder
	dec    *json.Decoder
	apiKey string
	model  string
}

/*
NewCohereProvider creates a new Cohere provider with the given API key and endpoint.
If apiKey is empty, it will try to read from the COHERE_API_KEY environment variable.
*/
func NewCohereProvider(
	apiKey string,
	endpoint string,
) *CohereProvider {
	errnie.Debug("provider.NewCohereProvider")

	if apiKey == "" {
		apiKey = os.Getenv("COHERE_API_KEY")
	}

	buf := bytes.NewBuffer([]byte{})
	buffer := bufio.NewReadWriter(
		bufio.NewReader(buf),
		bufio.NewWriter(buf),
	)

	// Create client
	cohereClient := cohereclient.NewClient(
		cohereclient.WithToken(apiKey),
		cohereclient.WithHTTPClient(
			&http.Client{
				Timeout: 5 * time.Second,
			},
		),
	)

	p := &CohereProvider{
		ProviderData: &ProviderData{
			Params: &ai.ContextData{},
			Result: &core.EventData{},
		},
		client: cohereClient,
		buffer: buffer,
		enc:    json.NewEncoder(buffer),
		dec:    json.NewDecoder(buffer),
		apiKey: apiKey,
		model:  "command", // Default to Command model
	}

	return p
}

/*
Read implements the io.Reader interface.
*/
func (provider *CohereProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.CohereProvider.Read")

	if err = provider.buffer.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if n, err = provider.buffer.Read(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	errnie.Debug("provider.CohereProvider.Read", "n", n, "err", err)
	return n, err
}

/*
Write implements the io.Writer interface.
*/
func (provider *CohereProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.CohereProvider.Write", "p", string(p))

	if n, err = provider.buffer.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = json.Unmarshal(p, provider.ProviderData.Params); err != nil {
		errnie.NewErrIO(err)
		return 0, err
	}

	errnie.Debug("provider.CohereProvider.Write", "n", n, "err", err)

	// Create chat message history from params
	messages := provider.buildMessages(provider.ProviderData.Params)

	// Configure the chat request
	toolsConfig := provider.buildTools(provider.ProviderData.Params)

	// Create the chat request parameters
	chatParams := &cohere.ChatStreamRequest{
		Model:       cohere.String(provider.model),
		Message:     "",
		ChatHistory: messages,
		Temperature: cohere.Float64(0.7),
		Tools:       toolsConfig,
	}

	// Add system prompt if present
	systemPrompt := provider.findSystemPrompt(provider.ProviderData.Params)
	if systemPrompt != "" {
		chatParams.Preamble = cohere.String(systemPrompt)
	}

	// Create streaming request handler
	err = errnie.NewErrIO(provider.handleStreamingRequest(chatParams))

	return n, err
}

/*
Close cleans up any resources.
*/
func (provider *CohereProvider) Close() error {
	errnie.Debug("provider.CohereProvider.Close")

	// Reset state
	provider.ProviderData.Params = nil
	provider.ProviderData.Result = nil

	provider.buffer = nil
	provider.enc = nil
	provider.dec = nil

	return nil
}

func (p *CohereProvider) findSystemPrompt(
	params *ai.ContextData,
) string {
	errnie.Debug("provider.findSystemPrompt")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "cohere")
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

func (p *CohereProvider) buildMessages(
	params *ai.ContextData,
) []*cohere.Message {
	errnie.Debug("provider.buildMessages")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "cohere")
		return nil
	}

	messages := make([]*cohere.Message, 0, len(params.Messages))

	for _, message := range params.Messages {
		// Skip system messages as they are handled separately via Preamble
		if message.Role == "system" {
			continue
		}

		switch message.Role {
		case "user":
			messages = append(messages, &cohere.Message{
				Role: "user",
				User: &cohere.ChatMessage{
					Message: message.Content,
				},
			})
		case "assistant":
			messages = append(messages, &cohere.Message{
				Role: "chatbot",
				Chatbot: &cohere.ChatMessage{
					Message: message.Content,
				},
			})
		default:
			errnie.Error("unknown message role", "role", message.Role)
			continue
		}
	}

	return messages
}

func (p *CohereProvider) buildTools(
	params *ai.ContextData,
) []*cohere.Tool {
	errnie.Debug("provider.buildTools")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "cohere")
		return nil
	}

	if len(params.Tools) == 0 {
		return nil
	}

	tools := make([]*cohere.Tool, 0, len(params.Tools))

	for _, tool := range params.Tools {
		// Create the tool configuration
		toolConfig := &cohere.Tool{
			Name:        tool.Name,
			Description: tool.Description,
		}

		tools = append(tools, toolConfig)
	}

	return tools
}

/*
handleStreamingRequest processes a streaming completion request
and emits chunks as they're received.
*/
func (provider *CohereProvider) handleStreamingRequest(
	params *cohere.ChatStreamRequest,
) (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	ctx := context.Background()

	// Create a streaming request
	stream, err := provider.client.ChatStream(ctx, params)

	if err != nil {
		errnie.Error("streaming error", "error", err)
		return errnie.NewErrHTTP(err, 500)
	}
	defer stream.Close()

	for {
		event, err := stream.Recv()

		if err == io.EOF {
			break
		}

		if err != nil {
			errnie.Error("streaming error", "error", err)
			return errnie.NewErrHTTP(err, 500)
		}

		// Check the event type and handle accordingly
		if content := event.TextGeneration.String(); content != "" {
			errnie.Debug("received stream chunk", "content", content)
			errnie.Debug("received stream chunk", "content", content)

			provider.Result = core.NewEvent(
				core.NewMessage(
					"assistant",
					"cohere",
					content,
				),
				nil,
			).EventData

			errnie.Debug("provider.handleStreamingRequest", "result", provider.Result)

			if err = provider.enc.Encode(provider.Result); err != nil {
				errnie.NewErrIO(err)
				return err
			}
		}
	}

	return nil
}

type CohereEmbedderData struct {
	Params *ai.ContextData `json:"params"`
	Result *[]float64      `json:"result"`
}

type CohereEmbedder struct {
	*CohereEmbedderData
	apiKey   string
	endpoint string
	client   *cohereclient.Client
	enc      *json.Encoder
	dec      *json.Decoder
	in       *bufio.ReadWriter
	out      *bufio.ReadWriter
}

func NewCohereEmbedder(apiKey string, endpoint string) *CohereEmbedder {
	errnie.Debug("provider.NewCohereEmbedder")

	if apiKey == "" {
		apiKey = os.Getenv("COHERE_API_KEY")
	}

	in := bufio.NewReadWriter(
		bufio.NewReader(bytes.NewBuffer([]byte{})),
		bufio.NewWriter(bytes.NewBuffer([]byte{})),
	)
	out := bufio.NewReadWriter(
		bufio.NewReader(bytes.NewBuffer([]byte{})),
		bufio.NewWriter(bytes.NewBuffer([]byte{})),
	)

	// Create client
	cohereClient := cohereclient.NewClient(
		cohereclient.WithToken(apiKey),
		cohereclient.WithHTTPClient(
			&http.Client{
				Timeout: 5 * time.Second,
			},
		),
	)

	embedder := &CohereEmbedder{
		CohereEmbedderData: &CohereEmbedderData{},
		apiKey:             apiKey,
		endpoint:           endpoint,
		client:             cohereClient,
		enc:                json.NewEncoder(out),
		dec:                json.NewDecoder(in),
		in:                 in,
		out:                out,
	}

	embedder.enc.Encode(embedder.CohereEmbedderData)

	return embedder
}

func (embedder *CohereEmbedder) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.CohereEmbedder.Read", "p", string(p))

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

func (embedder *CohereEmbedder) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.CohereEmbedder.Write")

	if n, err = embedder.in.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = embedder.in.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = embedder.dec.Decode(embedder.CohereEmbedderData); err != nil {
		errnie.NewErrIO(err)
		return
	}

	// Use Cohere's embedding API
	ctx := context.Background()

	// Get the text to embed
	var textToEmbed string
	if len(embedder.CohereEmbedderData.Params.Messages) > 0 {
		textToEmbed = embedder.CohereEmbedderData.Params.Messages[0].Content
	}

	in := cohere.EmbedInputType(cohere.EmbedInputTypeSearchDocument)

	// Create embedding request
	embedRequest := cohere.EmbedRequest{
		Texts:     []string{textToEmbed},
		Model:     cohere.String("embed-english-v3.0"), // Using a standard embedding model
		InputType: &in,
	}

	// Request embeddings
	embedResponse, err := embedder.client.Embed(ctx, &embedRequest)

	if err != nil {
		errnie.Error("embedding request failed", "error", err)
		*embedder.CohereEmbedderData.Result = make([]float64, 0)
	} else if embedResponse != nil && len(embedResponse.EmbeddingsFloats.Embeddings) > 0 {
		// Extract embeddings from response
		embeddings := embedResponse.EmbeddingsFloats.Embeddings

		// Convert to []float64
		float64Values := make([]float64, len(embeddings))

		// Store the result
		*embedder.CohereEmbedderData.Result = float64Values

		errnie.Debug("created embeddings",
			"text_length", len(textToEmbed),
			"dimensions", len(float64Values),
		)
	} else {
		errnie.Error("embedding response was empty or invalid")
		*embedder.CohereEmbedderData.Result = make([]float64, 0)
	}

	if err = embedder.enc.Encode(embedder.CohereEmbedderData); err != nil {
		errnie.NewErrIO(err)
		return
	}

	return len(p), nil
}

func (embedder *CohereEmbedder) Close() error {
	errnie.Debug("provider.CohereEmbedder.Close")

	embedder.CohereEmbedderData.Params = nil
	embedder.CohereEmbedderData.Result = nil

	return nil
}
