package errnie

import (
	"errors"
	"fmt"
	"strings"
)

type ErrnieError struct {
	msg string
}

func NewError(err error) error {
	if err == nil {
		return nil
	}

	return &ErrnieError{
		msg: Error(err).Error(),
	}
}

func (err *ErrnieError) Error() string { return err.msg }

type ErrIO struct{ Err error }

func NewErrIO(err error) error  { return &ErrIO{Err: err} }
func (err ErrIO) Error() string { return err.Err.Error() }

type ErrValidation struct{ Err error }

func NewErrValidation(args ...string) error {
	return &ErrValidation{Err: errors.New(strings.Join(args, " "))}
}

func (err ErrValidation) Error() string { return err.Err.Error() }

type ErrParse struct{ Err error }

func NewErrParse(err error) error  { return &ErrParse{Err: err} }
func (err ErrParse) Error() string { return err.Err.Error() }

type ErrOperation struct{ Err error }

func NewErrOperation(err error) error  { return &ErrOperation{Err: err} }
func (err ErrOperation) Error() string { return err.Err.Error() }

type ErrHTTP struct {
	Err  error
	Code int
}

func NewErrHTTP(err error, code int) error { return &ErrHTTP{Err: err, Code: code} }
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
