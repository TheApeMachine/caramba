package tools

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/theapemachine/caramba/pkg/core"
)

/*
ErrDivisionByZero is returned when a division operation attempts to divide by zero
*/
type ErrDivisionByZero struct {
	Message string
}

func (e *ErrDivisionByZero) Error() string {
	return e.Message
}

/*
ErrUnsupportedOperation is returned when the requested operation is not supported
*/
type ErrUnsupportedOperation struct {
	Message string
}

func (e *ErrUnsupportedOperation) Error() string {
	return e.Message
}

/*
CalculatorTool implements a basic calculator providing arithmetic operations
through the io.ReadWriteCloser interface
*/
type CalculatorTool struct {
	*core.BaseComponent
	Operation string  `json:"operation"`
	A         float64 `json:"a"`
	B         float64 `json:"b"`
}

/*
NewCalculatorTool creates a new calculator tool with its base component initialized
*/
func NewCalculatorTool() *CalculatorTool {
	return &CalculatorTool{
		BaseComponent: core.NewBaseComponent("calculator", core.TypeTool),
	}
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

	switch tool.Operation {
	case "add":
		result = tool.A + tool.B
	case "subtract":
		result = tool.A - tool.B
	case "multiply":
		result = tool.A * tool.B
	case "divide":
		if tool.B == 0 {
			return 0, &ErrDivisionByZero{Message: "division by zero"}
		}

		result = tool.A / tool.B
	default:
		return 0, &ErrUnsupportedOperation{
			Message: fmt.Sprintf("unsupported operation: %s", tool.Operation),
		}
	}

	response := map[string]interface{}{
		"result": result,
	}

	if responseBytes, err = json.Marshal(response); err != nil {
		return 0, err
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
	if err = json.Unmarshal(p, tool); err != nil {
		return 0, err
	}

	return len(p), nil
}

/*
Close resets the tool to its initial state
*/
func (tool *CalculatorTool) Close() error {
	tool.Operation = ""
	tool.A = 0
	tool.B = 0
	return nil
}
