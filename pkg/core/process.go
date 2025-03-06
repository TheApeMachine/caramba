package core

import (
	"encoding/json"
	"io"
)

/*
Process represents a structured output definition that can be used
to enforce specific response formats from an LLM using JSON Schema.
*/
type Process struct {
	*BaseComponent
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Schema      interface{} `json:"schema"`
}

/*
NewProcess creates a new Process with the given name, description, and schema.
*/
func NewProcess(name, description string, schema interface{}) *Process {
	return &Process{
		BaseComponent: NewBaseComponent(name, TypeProcess),
		Name:          name,
		Description:   description,
		Schema:        schema,
	}
}

/*
Read serializes the process definition to JSON and writes it to the provided buffer.
*/
func (p *Process) Read(buf []byte) (n int, err error) {
	data, err := json.Marshal(p)
	if err != nil {
		return 0, err
	}

	n = copy(buf, data)
	if n < len(data) {
		return n, io.ErrShortBuffer
	}

	return n, io.EOF
}

/*
Write updates the process definition from JSON data.
*/
func (p *Process) Write(data []byte) (n int, err error) {
	if err = json.Unmarshal(data, p); err != nil {
		return 0, err
	}

	return len(data), nil
}

/*
Close performs any necessary cleanup.
*/
func (p *Process) Close() error {
	return nil
}
