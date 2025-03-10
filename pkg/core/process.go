package core

import (
	"bytes"
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type ProcessData struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Schema      interface{} `json:"schema"`
}

/*
Process represents a structured output definition that can be used
to enforce specific response formats from an LLM using JSON Schema.
*/
type Process struct {
	*ProcessData
	dec *json.Decoder
	enc *json.Encoder
	in  *bytes.Buffer
	out *bytes.Buffer
}

/*
NewProcess creates a new Process with the given name, description, and schema.
*/
func NewProcess(name, description string, schema any) *Process {
	errnie.Debug("NewProcess")

	in := bytes.NewBuffer([]byte{})
	out := bytes.NewBuffer([]byte{})

	proc := &Process{
		ProcessData: &ProcessData{
			Name:        name,
			Description: description,
			Schema:      schema,
		},
		dec: json.NewDecoder(in),
		enc: json.NewEncoder(out),
		in:  in,
		out: out,
	}

	// Pre-encode the process data to JSON for reading
	proc.enc.Encode(proc.ProcessData)

	return proc
}

/*
Read serializes the process definition to JSON and writes it to the provided buffer.
*/
func (process *Process) Read(p []byte) (n int, err error) {
	errnie.Debug("Process.Read", "p", string(p))

	if process.out.Len() == 0 {
		return 0, io.EOF
	}

	return process.out.Read(p)
}

/*
Write updates the process definition from JSON data.
It implements the io.Writer interface for compatibility with io.Copy.
*/
func (process *Process) Write(p []byte) (n int, err error) {
	errnie.Debug("Process.Write", "p", string(p))

	// Reset the output buffer whenever we write new data
	if process.out.Len() > 0 {
		process.out.Reset()
	}

	// Write the incoming bytes to the input buffer
	n, err = process.in.Write(p)
	if err != nil {
		return n, err
	}

	// Try to decode the data from the input buffer
	// If it fails, we still return the bytes written but keep the error
	var buf ProcessData
	if decErr := process.dec.Decode(&buf); decErr == nil {
		// Only update if decoding was successful
		process.ProcessData.Name = buf.Name
		process.ProcessData.Description = buf.Description
		process.ProcessData.Schema = buf.Schema

		// Re-encode to the output buffer for subsequent reads
		if encErr := process.enc.Encode(process.ProcessData); encErr != nil {
			return n, errnie.NewErrIO(encErr)
		}
	}

	return n, nil
}

/*
Close performs any necessary cleanup.
*/
func (process *Process) Close() error {
	errnie.Debug("Process.Close")

	process.ProcessData.Name = ""
	process.ProcessData.Description = ""
	process.ProcessData.Schema = nil

	return nil
}
