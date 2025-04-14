package errors

// JSONRPCError represents a base JSON-RPC error
type JSONRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

func (e *JSONRPCError) Error() string {
	return e.Message
}

// JSONParseError represents an error parsing JSON payload
type JSONParseError struct {
	JSONRPCError
}

func NewJSONParseError() *JSONParseError {
	return &JSONParseError{
		JSONRPCError: JSONRPCError{
			Code:    -32700,
			Message: "Invalid JSON payload",
		},
	}
}

// InvalidRequestError represents a request payload validation error
type InvalidRequestError struct {
	JSONRPCError
}

func NewInvalidRequestError() *InvalidRequestError {
	return &InvalidRequestError{
		JSONRPCError: JSONRPCError{
			Code:    -32600,
			Message: "Request payload validation error",
		},
	}
}

// MethodNotFoundError represents an error when method is not found
type MethodNotFoundError struct {
	JSONRPCError
}

func NewMethodNotFoundError() *MethodNotFoundError {
	return &MethodNotFoundError{
		JSONRPCError: JSONRPCError{
			Code:    -32601,
			Message: "Method not found",
			Data:    nil,
		},
	}
}

// InvalidParamsError represents an error with invalid parameters
type InvalidParamsError struct {
	JSONRPCError
}

func NewInvalidParamsError() *InvalidParamsError {
	return &InvalidParamsError{
		JSONRPCError: JSONRPCError{
			Code:    -32602,
			Message: "Invalid parameters",
		},
	}
}

// InternalError represents an internal server error
type InternalError struct {
	JSONRPCError
}

func NewInternalError() *InternalError {
	return &InternalError{
		JSONRPCError: JSONRPCError{
			Code:    -32603,
			Message: "Internal error",
		},
	}
}

// TaskNotFoundError represents an error when task is not found
type TaskNotFoundError struct {
	JSONRPCError
}

func NewTaskNotFoundError() *TaskNotFoundError {
	return &TaskNotFoundError{
		JSONRPCError: JSONRPCError{
			Code:    -32001,
			Message: "Task not found",
			Data:    nil,
		},
	}
}

// TaskNotCancelableError represents an error when task cannot be canceled
type TaskNotCancelableError struct {
	JSONRPCError
}

func NewTaskNotCancelableError() *TaskNotCancelableError {
	return &TaskNotCancelableError{
		JSONRPCError: JSONRPCError{
			Code:    -32002,
			Message: "Task cannot be canceled",
			Data:    nil,
		},
	}
}

// PushNotificationNotSupportedError represents an error when push notifications are not supported
type PushNotificationNotSupportedError struct {
	JSONRPCError
}

func NewPushNotificationNotSupportedError() *PushNotificationNotSupportedError {
	return &PushNotificationNotSupportedError{
		JSONRPCError: JSONRPCError{
			Code:    -32003,
			Message: "Push Notification is not supported",
			Data:    nil,
		},
	}
}

// UnsupportedOperationError represents an error when operation is not supported
type UnsupportedOperationError struct {
	JSONRPCError
}

func NewUnsupportedOperationError() *UnsupportedOperationError {
	return &UnsupportedOperationError{
		JSONRPCError: JSONRPCError{
			Code:    -32004,
			Message: "This operation is not supported",
			Data:    nil,
		},
	}
}
