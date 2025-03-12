package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/url"

	sdk "github.com/ollama/ollama/api"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/utils"
)

/*
OllamaProvider implements an LLM provider that connects to Ollama's API.
It supports regular chat completions and streaming responses.
*/
type OllamaProvider struct {
	*ProviderData
	client *sdk.Client
	model  string
	buffer *bufio.ReadWriter
	enc    *json.Encoder
	dec    *json.Decoder
	cancel context.CancelFunc
}

/*
NewOllamaProvider creates a new Ollama provider with the given host endpoint.
If host is empty, it will try to read from configuration.
*/
func NewOllamaProvider(
	host string,
	model string,
) *OllamaProvider {
	errnie.Debug("provider.NewOllamaProvider")

	if host == "" {
		host = viper.GetViper().GetString("endpoints.ollama")
	}

	if model == "" {
		model = "llama3.2:3b" // Default model
	}

	hostURL, err := url.Parse(host)
	if err != nil {
		errnie.Error("failed to parse host URL", "error", err)
		return nil
	}

	buf := bytes.NewBuffer([]byte{})
	buffer := bufio.NewReadWriter(
		bufio.NewReader(buf),
		bufio.NewWriter(buf),
	)

	p := &OllamaProvider{
		ProviderData: &ProviderData{
			Params: &ai.ContextData{
				Messages: []*core.Message{},
			},
			Result: &core.Event{},
		},
		client: sdk.NewClient(hostURL, http.DefaultClient),
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
func (provider *OllamaProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.OllamaProvider.Read")

	if err = provider.buffer.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if n, err = provider.buffer.Read(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	errnie.Debug("provider.OllamaProvider.Read", "n", n, "err", err)
	return n, err
}

/*
Write implements the io.Writer interface.
*/
func (provider *OllamaProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.OllamaProvider.Write", "p", string(p))

	if n, err = provider.buffer.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if err = json.Unmarshal(p, provider.ProviderData.Params); err != nil {
		errnie.NewErrIO(err)
		return 0, err
	}

	errnie.Debug("provider.OllamaProvider.Write", "n", n, "err", err)

	// Create the Ollama request and handle streaming
	err = errnie.NewErrIO(provider.handleStreamingRequest())

	return n, err
}

/*
Close cleans up any resources.
*/
func (provider *OllamaProvider) Close() error {
	errnie.Debug("provider.OllamaProvider.Close")

	// Cancel any ongoing streaming
	if provider.cancel != nil {
		provider.cancel()
	}

	// Reset state
	provider.ProviderData.Params = nil
	provider.ProviderData.Result = nil

	provider.buffer = nil
	provider.enc = nil
	provider.dec = nil

	return nil
}

func (p *OllamaProvider) buildMessages(
	params *ai.ContextData,
) []sdk.Message {
	errnie.Debug("provider.buildMessages")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "ollama")
		return nil
	}

	messages := make([]sdk.Message, 0, len(params.Messages))

	for _, message := range params.Messages {
		if message.Content != "" {
			messages = append(messages, sdk.Message{
				Role:    message.Role,
				Content: message.Content,
			})
		}
	}

	return messages
}

func (p *OllamaProvider) buildTools(
	params *ai.ContextData,
) []sdk.Tool {
	errnie.Debug("provider.buildTools")

	if params == nil {
		errnie.NewErrValidation("params are nil", "provider", "ollama")
		return nil
	}

	tools := make([]sdk.Tool, 0, len(params.Tools))

	for _, tool := range params.Tools {
		// Create tool from our schema
		toolParam := sdk.Tool{
			Type: "function",
			Function: sdk.ToolFunction{
				Name:        tool.ToolData.Name,
				Description: tool.ToolData.Description,
				Parameters:  utils.GenerateSchema[core.Tool]().(sdk.ToolFunction).Parameters,
			},
		}

		tools = append(tools, toolParam)
	}

	return tools
}

/*
handleStreamingRequest processes a streaming completion request
and emits chunks as they're received.
*/
func (p *OllamaProvider) handleStreamingRequest() (err error) {
	errnie.Debug("provider.handleStreamingRequest")

	ctx, cancel := context.WithCancel(context.Background())
	p.cancel = cancel
	defer cancel()

	// Build the request messages from our context data
	messages := p.buildMessages(p.ProviderData.Params)
	if len(messages) == 0 {
		err = errnie.NewErrValidation("no valid messages to process", "provider", "ollama")
		return
	}

	stream := true

	// Build the request with non-empty fields
	request := &sdk.ChatRequest{
		Model:    p.model,
		Messages: messages,
		Stream:   &stream,
	}

	// Add tools if we have any
	tools := p.buildTools(p.ProviderData.Params)
	if len(tools) > 0 {
		request.Tools = sdk.Tools(tools)
	}

	errnie.Debug("streaming request initialized", "model", p.model)

	// Use Ollama's streaming API to process the response
	if err = p.client.Chat(ctx, request, func(resp sdk.ChatResponse) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			if resp.Message.Content != "" {
				errnie.Debug("received stream chunk",
					"content", resp.Message.Content,
				)

				p.Result = core.NewEvent(
					core.NewMessage(
						"assistant",
						"ollama",
						resp.Message.Content,
					),
					nil,
				)

				errnie.Debug("provider.handleStreamingRequest", "result", p.Result)

				if err = p.enc.Encode(p.Result); err != nil {
					errnie.NewErrIO(err)
					return err
				}
			}
			return nil
		}
	}); err != nil {
		return errnie.NewErrHTTP(err, 500)
	}

	return err
}
