package core

import (
	"encoding/json"
	"io"
)

/*
Tool represents a schema definition for an executable capability.
It serves as both a schema definition and can be connected to an
implementation with the WithFunction method.
*/
type Tool struct {
	*BaseComponent
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  []Parameter `json:"parameters"`
	Strict      bool        `json:"strict"`
	fn          io.ReadWriteCloser
}

/*
Parameter defines a specific input that a tool accepts, including
type information and validation constraints.
*/
type Parameter struct {
	Type                 string              `json:"type"`
	Properties           map[string]Property `json:"properties"`
	Required             []string            `json:"required"`
	AdditionalProperties bool                `json:"additionalProperties"`
}

/*
Property describes a single field within a parameter, including
its type and description.
*/
type Property struct {
	Type        string `json:"type"`
	Description string `json:"description"`
	Enum        []any  `json:"enum"`
}

/*
NewTool creates a tool schema with the given metadata.
*/
func NewTool(
	name string,
	description string,
	parameters []Parameter,
) *Tool {
	return &Tool{
		BaseComponent: NewBaseComponent(name, TypeTool),
		Name:          name,
		Description:   description,
		Parameters:    parameters,
		Strict:        true,
	}
}

/*
Read serializes the tool schema to JSON and writes it to the provided buffer.
This allows tools to describe themselves when read from.
*/
func (tool *Tool) Read(p []byte) (n int, err error) {
	var data []byte

	if data, err = json.Marshal(tool); err != nil {
		return 0, err
	}

	n = copy(p, data)

	if n < len(data) {
		return n, io.ErrShortBuffer
	}

	return n, io.EOF
}

/*
Write accepts a JSON representation of tool parameters and updates the tool's state.
*/
func (tool *Tool) Write(p []byte) (n int, err error) {
	if err = json.Unmarshal(p, tool); err != nil {
		return 0, err
	}

	return len(p), nil
}

/*
Close cleans up any resources associated with the tool's implementation.
*/
func (tool *Tool) Close() error {
	if tool.fn != nil {
		return tool.fn.Close()
	}

	return nil
}

/*
WithFunction connects a concrete implementation to this tool schema.
This allows a tool schema to be created separately from its implementation.
*/
func (tool *Tool) WithFunction(fn io.ReadWriteCloser) *Tool {
	tool.fn = fn
	return tool
}
