package core

import (
	"bytes"
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type ToolCallData struct {
	ID        string         `json:"id"`
	ToolName  string         `json:"tool_name"`
	Arguments map[string]any `json:"arguments"`
}

type ToolCall struct {
	*ToolCallData
	dec *json.Decoder
	enc *json.Encoder
	in  *bytes.Buffer
	out *bytes.Buffer
}

func NewToolCall(id string, toolName string, arguments map[string]any) *ToolCall {
	in := bytes.NewBuffer([]byte{})
	out := bytes.NewBuffer([]byte{})

	tc := &ToolCall{
		ToolCallData: &ToolCallData{
			ID:        id,
			ToolName:  toolName,
			Arguments: arguments,
		},
		dec: json.NewDecoder(in),
		enc: json.NewEncoder(out),
		in:  in,
		out: out,
	}

	// Pre-encode the tool call to JSON for reading
	tc.enc.Encode(tc.ToolCallData)

	return tc
}

func (toolcall *ToolCall) Read(p []byte) (n int, err error) {
	if toolcall.out.Len() == 0 {
		if err = errnie.NewErrIO(toolcall.enc.Encode(toolcall.ToolCallData)); err != nil {
			return 0, err
		}
	}

	return toolcall.out.Read(p)
}

func (toolcall *ToolCall) Write(p []byte) (n int, err error) {
	// Reset the output buffer whenever we write new data
	if toolcall.out.Len() > 0 {
		toolcall.out.Reset()
	}

	// Write the incoming bytes to the input buffer
	n, err = toolcall.in.Write(p)
	if err != nil {
		return n, err
	}

	// Try to decode the data from the input buffer
	// If it fails, we still return the bytes written but keep the error
	var buf ToolCallData
	if decErr := toolcall.dec.Decode(&buf); decErr == nil {
		// Only update if decoding was successful
		toolcall.ToolCallData.ID = buf.ID
		toolcall.ToolCallData.ToolName = buf.ToolName
		toolcall.ToolCallData.Arguments = buf.Arguments

		// Re-encode to the output buffer for subsequent reads
		if encErr := toolcall.enc.Encode(toolcall.ToolCallData); encErr != nil {
			return n, errnie.NewErrIO(encErr)
		}
	}

	return n, nil
}

func (toolcall *ToolCall) Close() error {
	return nil
}

func (toolcall *ToolCall) Name() string {
	return toolcall.ToolCallData.ToolName
}

func (toolcall *ToolCall) Arguments() map[string]any {
	return toolcall.ToolCallData.Arguments
}

func (toolcall *ToolCall) ID() string {
	return toolcall.ToolCallData.ID
}

type ToolData struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  []Parameter `json:"parameters"`
	Strict      bool        `json:"strict"`
}

/*
Tool represents a schema definition for an executable capability.
It serves as both a schema definition and can be connected to an
implementation with the WithFunction method.
*/
type Tool struct {
	*ToolData
	dec *json.Decoder
	enc *json.Encoder
	in  *bytes.Buffer
	out *bytes.Buffer
	fn  io.ReadWriteCloser
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
	in := bytes.NewBuffer([]byte{})
	out := bytes.NewBuffer([]byte{})

	return &Tool{
		ToolData: &ToolData{
			Name:        name,
			Description: description,
			Parameters:  parameters,
			Strict:      true,
		},
		dec: json.NewDecoder(in),
		enc: json.NewEncoder(out),
		in:  in,
		out: out,
	}
}

/*
Read serializes the tool schema to JSON and writes it to the provided buffer.
This allows tools to describe themselves when read from.
*/
func (tool *Tool) Read(p []byte) (n int, err error) {
	if tool.out.Len() == 0 {
		if err = errnie.NewErrIO(tool.enc.Encode(tool.ToolData)); err != nil {
			return 0, err
		}
	}

	return tool.out.Read(p)
}

/*
Write accepts a JSON representation of tool parameters and updates the tool's state.
*/
func (tool *Tool) Write(p []byte) (n int, err error) {
	// Reset the output buffer whenever we write new data
	if tool.out.Len() > 0 {
		tool.out.Reset()
	}

	// Write the incoming bytes to the input buffer
	n, err = tool.in.Write(p)
	if err != nil {
		return n, err
	}

	// Try to decode the data from the input buffer
	// If it fails, we still return the bytes written but keep the error
	var buf ToolData
	if decErr := tool.dec.Decode(&buf); decErr == nil {
		// Only update if decoding was successful
		tool.ToolData.Name = buf.Name
		tool.ToolData.Description = buf.Description
		tool.ToolData.Parameters = buf.Parameters
		tool.ToolData.Strict = buf.Strict

		// Re-encode to the output buffer for subsequent reads
		if encErr := tool.enc.Encode(tool.ToolData); encErr != nil {
			return n, errnie.NewErrIO(encErr)
		}
	}

	return n, nil
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

func (tool *Tool) Function() io.ReadWriteCloser {
	return tool.fn
}

func (tool *Tool) Name() string {
	return tool.ToolData.Name
}

func (tool *Tool) Description() string {
	return tool.ToolData.Description
}

func (tool *Tool) Parameters() []Parameter {
	return tool.ToolData.Parameters
}

func (tool *Tool) Strict() bool {
	return tool.ToolData.Strict
}
