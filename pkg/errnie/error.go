package errnie

import (
	"errors"
	"fmt"
	"io"
	"strings"
)

type ErrnieErrorType string

const (
	ErrTypeUnknown    ErrnieErrorType = "unknown"
	ErrTypeIO         ErrnieErrorType = "io"
	ErrTypeValidation ErrnieErrorType = "validation"
	ErrTypeParse      ErrnieErrorType = "parse"
	ErrTypeOperation  ErrnieErrorType = "operation"
	ErrTypeHTTP       ErrnieErrorType = "http"
)

var (
	ErrInvalidInput = errors.New("invalid input")
)

func Unwrap(err error) *ErrnieError {
	if err == nil {
		return nil
	}

	if _, ok := err.(*ErrnieError); !ok {
		// If the error is not an ErrnieError, wrap it in an ErrnieError.
		return Unwrap(NewError(err))
	}

	return err.(*ErrnieError)
}

func (err *ErrnieError) Unwrap() error {
	return err.Errors[0]
}

/*
ErrnieError implement the error interface, while also providing a way to
accumulate multiple errors transparently.
*/
type ErrnieError struct {
	Errors          []error         `json:"errors"`
	Msg             string          `json:"msg"`
	ErrnieErrorType ErrnieErrorType `json:"errnieErrorType"`
}

func NewError(err error) error {
	if err == nil {
		return nil
	}

	// Log the error.
	Error(err)

	return &ErrnieError{
		Errors: []error{err},
	}
}

func (err *ErrnieError) WithType(t ErrnieErrorType) *ErrnieError {
	err.ErrnieErrorType = t
	return err
}

func (err *ErrnieError) Error() string { return err.Msg }

func (err *ErrnieError) Is(t ErrnieErrorType) bool {
	return err.ErrnieErrorType == t
}

func (err *ErrnieError) Add(errs ...error) error {
	// If there are no errors, return nil.
	if len(errs) == 0 {
		return nil
	}

	// Add the errors to the list.
	err.Errors = append(err.Errors, errs...)

	for _, e := range errs {
		// Log the error.
		Error(e)
	}

	return err
}

type ErrIO struct{ Err error }

func NewErrIO(err error) error {
	if err == nil {
		return nil
	}

	if errors.Is(err, io.EOF) {
		return err
	}

	return &ErrIO{Err: NewError(err)}
}

func (err ErrIO) Error() string { return err.Err.Error() }

type ErrValidation struct{ Err error }

func NewErrValidation(args ...string) error {
	if len(args) == 0 {
		return nil
	}

	return &ErrValidation{Err: NewError(errors.New(strings.Join(args, " ")))}
}

func (err ErrValidation) Error() string { return err.Err.Error() }

type ErrParse struct{ Err error }

func NewErrParse(err error) error {
	if err == nil || errors.Is(err, io.EOF) {
		return nil
	}

	return &ErrParse{Err: NewError(err)}
}

func (err ErrParse) Error() string { return err.Err.Error() }

type ErrOperation struct{ Err error }

func NewErrOperation(err error) error {
	if err == nil || errors.Is(err, io.EOF) {
		return nil
	}

	return &ErrOperation{Err: NewError(err)}
}

func (err ErrOperation) Error() string { return err.Err.Error() }

type ErrHTTP struct {
	Err  error
	Code int
}

func NewErrHTTP(err error, code int) error {
	if err == nil || errors.Is(err, io.EOF) {
		return nil
	}

	return &ErrHTTP{Err: NewError(err), Code: code}
}

func (err ErrHTTP) Error() string {
	switch err.Code {
	case 400:
		return fmt.Sprintf("Bad Request: %s", err.Err.Error())
	case 401:
		return fmt.Sprintf("Unauthorized: %s", err.Err.Error())
	case 403:
		return fmt.Sprintf("Forbidden: %s", err.Err.Error())
	case 404:
		return fmt.Sprintf("Not Found: %s", err.Err.Error())
	case 429:
		return fmt.Sprintf("Too Many Requests: %s", err.Err.Error())
	case 500:
		return fmt.Sprintf("Internal Server Error: %s", err.Err.Error())
	case 502:
		return fmt.Sprintf("Bad Gateway: %s", err.Err.Error())
	case 503:
		return fmt.Sprintf("Service Unavailable: %s", err.Err.Error())
	case 504:
		return fmt.Sprintf("Gateway Timeout: %s", err.Err.Error())
	default:
		return fmt.Sprintf("HTTP Error: %d %s", err.Code, err.Err.Error())
	}
}
