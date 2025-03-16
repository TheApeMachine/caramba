package utils

import (
	"bytes"
	"encoding/gob"
	"errors"

	"github.com/theapemachine/caramba/pkg/errnie"
)

// APIError represents the structure of OpenAI API errors.
// This mirrors the structure of the internal apierror.Error type.
type APIError struct {
	Message string      `json:"message"`
	Type    string      `json:"type"`
	Param   interface{} `json:"param"`
	Code    interface{} `json:"code"`
}

// Error makes APIError implement the error interface
func (e *APIError) Error() string {
	return e.Message
}

// Additional API error types that might be used
type RequestError struct {
	APIError
}

type AuthenticationError struct {
	APIError
}

type PermissionError struct {
	APIError
}

type RateLimitError struct {
	APIError
}

type ServerError struct {
	APIError
}

type InvalidRequestError struct {
	APIError
}

// init registers types that might be serialized via interfaces.
func init() {
	// Register error types used in the application
	gob.Register(errnie.ErrIO{})
	gob.Register(errnie.ErrValidation{})
	gob.Register(errnie.ErrParse{})
	gob.Register(errnie.ErrOperation{})
	gob.Register(errnie.ErrHTTP{})
	gob.Register(&errnie.ErrnieError{})
	gob.Register(errors.New(""))

	// Register OpenAI API error types - only register the pointer version
	// to avoid duplicate registration
	gob.Register(&APIError{})
	gob.Register(&RequestError{})
	gob.Register(&AuthenticationError{})
	gob.Register(&PermissionError{})
	gob.Register(&RateLimitError{})
	gob.Register(&ServerError{})
	gob.Register(&InvalidRequestError{})
}

func GobEncode(md any) ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	if err := enc.Encode(md); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}
