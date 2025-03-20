package errnie

import (
	"net/http"
	"slices"
	"strings"
)

type ErrnieError struct {
	errors   []error
	messages []string
	status   ErrnieStatusType
}

func (e *ErrnieError) Error() string {
	return strings.Join(e.messages, " ")
}

func (e *ErrnieError) Status() int {
	switch e.status {
	case ErrnieStatusTypeOK:
		return http.StatusOK
	case ErrnieStatusTypeBadRequest:
		return http.StatusBadRequest
	case ErrnieStatusTypeUnauthorized:
		return http.StatusUnauthorized
	case ErrnieStatusTypeForbidden:
		return http.StatusForbidden
	case ErrnieStatusTypeNotFound:
		return http.StatusNotFound
	default:
		return http.StatusInternalServerError
	}
}
func (e *ErrnieError) Unwrap() error {
	if len(e.errors) > 0 {
		return e.errors[0]
	}
	return nil
}

func (e *ErrnieError) Is(target error) bool {
	return slices.Contains(e.errors, target)
}

type ErrnieErrorOption func(*ErrnieError)

func New(options ...ErrnieErrorOption) *ErrnieError {
	err := &ErrnieError{
		errors:   make([]error, 0),
		messages: make([]string, 0),
		status:   0,
	}

	for _, option := range options {
		option(err)
	}

	if len(err.errors) > 0 {
		Error(err.errors[len(err.errors)-1])
	}

	return err
}

func NotFound(err error, msg ...string) *ErrnieError {
	return New(
		WithStatus(ErrnieStatusTypeNotFound),
		WithError(err),
		WithMessage(strings.Join(msg, " ")),
	)
}

func BadRequest(err error, msg ...string) *ErrnieError {
	return New(
		WithStatus(ErrnieStatusTypeBadRequest),
		WithError(err),
		WithMessage(strings.Join(msg, " ")),
	)
}

func InternalError(err error, msg ...string) *ErrnieError {
	return New(
		WithStatus(ErrnieStatusTypeInternalServerError),
		WithError(err),
		WithMessage(strings.Join(msg, " ")),
	)
}

func WithError(err error) ErrnieErrorOption {
	return func(e *ErrnieError) {
		e.errors = append(e.errors, err)
	}
}

func WithMessage(message string) ErrnieErrorOption {
	return func(e *ErrnieError) {
		e.messages = append(e.messages, message)
	}
}

func WithStatus(status ErrnieStatusType) ErrnieErrorOption {
	return func(e *ErrnieError) {
		e.status = status
	}
}

type ErrnieErrorType uint

const (
	ErrnieErrorTypeUnknown ErrnieErrorType = iota
	ErrnieErrorTypeIO
	ErrnieErrorTypeBadRequest
	ErrnieErrorTypeUnauthorized
	ErrnieErrorTypeForbidden
	ErrnieErrorTypeNotFound
)

func (et *ErrnieErrorType) Error() string {
	switch *et {
	case ErrnieErrorTypeUnknown:
		return "unknown"
	case ErrnieErrorTypeIO:
		return "io"
	case ErrnieErrorTypeBadRequest:
		return "bad request"
	case ErrnieErrorTypeUnauthorized:
		return "unauthorized"
	case ErrnieErrorTypeForbidden:
		return "forbidden"
	case ErrnieErrorTypeNotFound:
		return "not found"
	default:
		return "undefined"
	}
}

type ErrnieStatusType uint

const (
	ErrnieStatusTypeUnknown ErrnieStatusType = iota
	ErrnieStatusTypeOK
	ErrnieStatusTypeBadRequest
	ErrnieStatusTypeUnauthorized
	ErrnieStatusTypeForbidden
	ErrnieStatusTypeNotFound
	ErrnieStatusTypeInternalServerError
)
