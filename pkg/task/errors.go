package task

import (
	"fmt"
)

// NewInvalidParamsError creates a TaskRequestError for invalid parameters (-32602).
func NewInvalidParamsError(method string, err error) *TaskRequestError {
	data := ""
	if err != nil {
		data = err.Error()
	}
	return &TaskRequestError{
		Code:    -32602, // JSON-RPC Invalid params
		Message: fmt.Sprintf("Invalid params for %s", method),
		Data:    data,
	}
}

// NewMissingParamError creates a TaskRequestError for a missing required parameter (-32602).
func NewMissingParamError(method string, paramName string) *TaskRequestError {
	return &TaskRequestError{
		Code:    -32602, // JSON-RPC Invalid params
		Message: fmt.Sprintf("Invalid params for %s", method),
		Data:    fmt.Sprintf("Missing required parameter: %s", paramName),
	}
}

// NewTaskNotFoundError creates a TaskRequestError when a task is not found (-32001).
func NewTaskNotFoundError(taskID string, err error) *TaskRequestError {
	data := fmt.Sprintf("Task with ID '%s' not found", taskID)
	if err != nil {
		data = fmt.Sprintf("%s or error retrieving: %s", data, err.Error())
	}
	return &TaskRequestError{
		Code:    -32001, // Application-defined: Task not found
		Message: "Task not found",
		Data:    data,
	}
}

// NewTaskCannotBeCanceledError creates a TaskRequestError when a task is in a final state (-32002).
func NewTaskCannotBeCanceledError(taskID string, currentState TaskState) *TaskRequestError {
	return &TaskRequestError{
		Code:    -32002, // Application-defined: Task cannot be canceled
		Message: "Task cannot be canceled",
		Data:    fmt.Sprintf("Task '%s' is already in a final state ('%s')", taskID, currentState.String()),
	}
}

// NewCapabilityError creates a TaskRequestError when an agent lacks a required capability (-32003).
func NewCapabilityError(method string, capability string) *TaskRequestError {
	return &TaskRequestError{
		Code:    -32003, // Application-defined: Capability required
		Message: "Capability required",
		Data:    fmt.Sprintf("Agent lacks required capability '%s' for method '%s'", capability, method),
	}
}

// NewInternalError creates a generic internal TaskRequestError (-32603 or -32000).
// Use -32603 for general server errors during request processing, -32000 for application-specific internal issues.
func NewInternalError(method string, message string, err error, code int) *TaskRequestError {
	data := ""
	if err != nil {
		data = err.Error()
	}
	// Default to -32000 if an invalid code is passed, otherwise use provided code.
	finalCode := code
	if finalCode > -32000 || finalCode < -32700 {
		finalCode = -32000 // Application-defined server error
	}

	return &TaskRequestError{
		Code:    finalCode,
		Message: fmt.Sprintf("Internal error in %s: %s", method, message),
		Data:    data,
	}
}

// NewLLMError creates a TaskRequestError specific to LLM interaction failures (-32000 or -32003).
func NewLLMError(method string, message string, err error) *TaskRequestError {
	// Use -32003 if it's about capability/setup, -32000 for general processing.
	// Let's default to -32000 for now unless we have more context.
	code := -32000
	return NewInternalError(method, fmt.Sprintf("LLM Error: %s", message), err, code)
}

// NewIncompatibleContentTypesError creates a TaskRequestError for content type mismatches.
func NewIncompatibleContentTypesError(methodName string, clientType, agentType string) *TaskRequestError {
	return &TaskRequestError{
		Code:    -32005,
		Message: "Incompatible content types",
		Data:    fmt.Sprintf("Method %s: Client content type '%s' is incompatible with agent content type '%s'", methodName, clientType, agentType),
	}
}
