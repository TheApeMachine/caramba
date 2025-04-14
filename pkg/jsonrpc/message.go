package jsonrpc

import (
	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/errors"
)

// Message represents a base JSON-RPC message
type Message struct {
	JSONRPC string `json:"jsonrpc"`
	ID      string `json:"id"`
}

// NewMessage creates a new JSON-RPC message
func NewMessage() Message {
	return Message{
		JSONRPC: "2.0",
		ID:      uuid.New().String(),
	}
}

// Request represents a JSON-RPC request
type Request struct {
	Message
	Method string `json:"method"`
	Params any    `json:"params,omitempty"`
}

// NewRequest creates a new JSON-RPC request
func NewRequest(method string, params any) Request {
	return Request{
		Message: NewMessage(),
		Method:  method,
		Params:  params,
	}
}

// Response represents a JSON-RPC response
type Response struct {
	Message
	Result any                  `json:"result,omitempty"`
	Error  *errors.JSONRPCError `json:"error,omitempty"`
}

// NewResponse creates a new JSON-RPC response
func NewResponse(result any, err *errors.JSONRPCError) Response {
	return Response{
		Message: NewMessage(),
		Result:  result,
		Error:   err,
	}
}
