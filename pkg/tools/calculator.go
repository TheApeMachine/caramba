package tools

import (
	"bytes"
	"encoding/json"
	"errors"
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
	errnie.Debug("NewCalculatorTool")

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
Read reads the calculation result from the output buffer
*/
func (tool *CalculatorTool) Read(p []byte) (n int, err error) {
	errnie.Debug("CalculatorTool.Read", "p", string(p))

	if tool.out.Len() == 0 {
		return 0, io.EOF
	}

	return tool.out.Read(p)
}

/*
performCalculation performs the calculation defined by the tool's parameters
and encodes the result to the output buffer
*/
func (tool *CalculatorTool) performCalculation() error {
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
			return errnie.NewErrOperation(errors.New("division by zero"))
		}

		result = tool.CalculatorToolData.A / tool.CalculatorToolData.B
	default:
		return errnie.NewErrValidation("invalid operation")
	}

	// Format the result for output
	responseBytes, err := json.Marshal(map[string]interface{}{
		"result": result,
	})

	if err != nil {
		return errnie.NewErrParse(err)
	}

	// Reset the output buffer
	tool.out.Reset()

	// Write the result to the output buffer
	_, err = tool.out.Write(responseBytes)
	return errnie.NewErrIO(err)
}

/*
Write updates the calculator's parameters from JSON data.
*/
func (tool *CalculatorTool) Write(p []byte) (n int, err error) {
	errnie.Debug("CalculatorTool.Write", "p", string(p))

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

		// Perform the calculation and update the output buffer
		if err := tool.performCalculation(); err != nil {
			return n, err
		}
	}

	return n, nil
}

/*
Close resets the tool to its initial state
*/
func (tool *CalculatorTool) Close() error {
	errnie.Debug("CalculatorTool.Close")

	tool.CalculatorToolData.Operation = ""
	tool.CalculatorToolData.A = 0
	tool.CalculatorToolData.B = 0
	return nil
}

func (tool *CalculatorTool) WithFunction(fn io.ReadWriteCloser) *CalculatorTool {
	errnie.Debug("CalculatorTool.WithFunction")

	tool.fn = fn
	return tool
}
