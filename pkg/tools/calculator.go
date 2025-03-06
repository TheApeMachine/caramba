package tools

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type CalculatorToolData struct {
	Operation string  `json:"operation"`
	A         float64 `json:"a"`
	B         float64 `json:"b"`
}

/*
CalculatorTool implements a basic calculator providing arithmetic operations
through the io.ReadWriteCloser interface
*/
type CalculatorTool struct {
	*CalculatorToolData
	enc *json.Encoder
	dec *json.Decoder
	in  *bytes.Buffer
	out *bytes.Buffer
	fn  io.ReadWriteCloser
}

/*
NewCalculatorTool creates a new calculator tool with its base component initialized
*/
func NewCalculatorTool() *CalculatorTool {
	in := bytes.NewBuffer([]byte{})
	out := bytes.NewBuffer([]byte{})

	tool := &CalculatorTool{
		CalculatorToolData: &CalculatorToolData{},
		enc:                json.NewEncoder(out),
		dec:                json.NewDecoder(in),
		in:                 in,
		out:                out,
	}

	// Pre-encode the tool data to JSON for reading
	tool.enc.Encode(tool.CalculatorToolData)

	return tool
}

/*
Read performs the calculation defined by the tool's parameters
and returns the result as JSON
*/
func (tool *CalculatorTool) Read(p []byte) (n int, err error) {
	var (
		result        float64
		responseBytes []byte
	)

	switch tool.CalculatorToolData.Operation {
	case "add":
		result = tool.CalculatorToolData.A + tool.CalculatorToolData.B
	case "subtract":
		result = tool.CalculatorToolData.A - tool.CalculatorToolData.B
	case "multiply":
		result = tool.CalculatorToolData.A * tool.CalculatorToolData.B
	case "divide":
		if tool.CalculatorToolData.B == 0 {
			return 0, errnie.NewErrOperation(errors.New("division by zero"))
		}

		result = tool.CalculatorToolData.A / tool.CalculatorToolData.B
	default:
		return 0, errnie.NewErrOperation(fmt.Errorf("unsupported operation: %s", tool.CalculatorToolData.Operation))
	}

	response := map[string]interface{}{
		"result": result,
	}

	if responseBytes, err = json.Marshal(response); err != nil {
		return 0, errnie.NewErrIO(err)
	}

	n = copy(p, responseBytes)

	if n < len(responseBytes) {
		return n, io.ErrShortBuffer
	}

	return n, io.EOF
}

/*
Write accepts operation parameters as JSON and configures the tool
*/
func (tool *CalculatorTool) Write(p []byte) (n int, err error) {
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
	var buf CalculatorToolData
	if decErr := tool.dec.Decode(&buf); decErr == nil {
		// Only update if decoding was successful
		tool.CalculatorToolData.Operation = buf.Operation
		tool.CalculatorToolData.A = buf.A
		tool.CalculatorToolData.B = buf.B

		// Re-encode to the output buffer for subsequent reads
		if encErr := tool.enc.Encode(tool.CalculatorToolData); encErr != nil {
			return n, errnie.NewErrIO(encErr)
		}
	}

	return n, nil
}

/*
Close resets the tool to its initial state
*/
func (tool *CalculatorTool) Close() error {
	tool.CalculatorToolData.Operation = ""
	tool.CalculatorToolData.A = 0
	tool.CalculatorToolData.B = 0
	return nil
}

func (tool *CalculatorTool) WithFunction(fn io.ReadWriteCloser) *CalculatorTool {
	tool.fn = fn
	return tool
}
