package core

import (
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

// JSONComponent provides shared functionality for components that need JSON serialization
type JSONComponent struct {
	codec io.ReadWriteCloser
}

// NewJSONComponent creates a new component with JSON handling capabilities
func NewJSONComponent(codec io.ReadWriteCloser) *JSONComponent {
	return &JSONComponent{
		codec: codec,
	}
}

// ReadJSON reads and deserializes JSON data into the provided value
func (jc *JSONComponent) ReadJSON(v any) error {
	var rawData json.RawMessage

	if err := json.NewDecoder(jc.codec).Decode(&rawData); err != nil {
		return errnie.Error("failed to read JSON data", "error", err)
	}

	return json.Unmarshal(rawData, v)
}

// WriteJSON serializes and writes the provided value as JSON
func (jc *JSONComponent) WriteJSON(v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return errnie.Error("failed to marshal JSON data", "error", err)
	}

	_, err = jc.codec.Write(data)
	return errnie.Error("failed to write JSON data", "error", err)
}

// Close closes the underlying codec
func (jc *JSONComponent) Close() error {
	return jc.codec.Close()
}
