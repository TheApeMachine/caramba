package workflow

import (
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
)

type ProviderComponent struct {
	codec    *JSONCodec
	provider *provider.OpenAIProvider
}

func NewProviderComponent(provider *provider.OpenAIProvider, inputStream, outputStream *PipeStream) *ProviderComponent {
	return &ProviderComponent{
		codec:    NewJSONCodec(inputStream),
		provider: provider,
	}
}

func (pc *ProviderComponent) Read(p []byte) (int, error) {
	errnie.Debug("Reading from provider component")

	var v any
	if err := pc.codec.Read(&v); err != nil {
		errnie.Error("Failed to read from provider component", "error", err)
		return 0, err
	}

	data, err := json.Marshal(v)
	if err != nil {
		errnie.Error("Failed to marshal data", "error", err)
		return 0, err
	}

	if len(p) < len(data) {
		errnie.Error("Short buffer")
		return 0, io.ErrShortBuffer
	}

	return copy(p, data), nil
}

func (pc *ProviderComponent) Write(p []byte) (int, error) {
	errnie.Debug("Writing to provider component", "len", len(p))
	errnie.Debug(string(p))

	var context ai.Context
	if err := json.Unmarshal(p, &context); err != nil {
		errnie.Error("Failed to unmarshal context", "error", err)
		return 0, err
	}

	errnie.Debug("Provider received context")

	result, err := pc.provider.GenerateResponse(&context)
	if err != nil {
		errnie.Error("Failed to generate response", "error", err)
		return 0, err
	}

	resultBytes, err := json.Marshal(result)
	if err != nil {
		errnie.Error("Failed to marshal result", "error", err)
		return 0, err
	}

	errnie.Debug("Provider writing result")

	if err := pc.codec.Write(resultBytes); err != nil {
		errnie.Error("Failed to write result", "error", err)
		return 0, err
	}

	errnie.Debug("Provider successfully wrote result")

	return len(resultBytes), nil
}

func (pc *ProviderComponent) Close() error {
	return pc.codec.Close()
}
