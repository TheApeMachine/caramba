package stream

import (
	"errors"
	"fmt"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type BufferIOError struct {
	err   error
	scope string
}

func NewBufferIOError(scope string, err error) *BufferIOError {
	if err == nil || err == io.EOF {
		return nil
	}

	return &BufferIOError{
		err:   errnie.Error(fmt.Errorf("%s - %w", scope, err)),
		scope: scope,
	}
}

func (e *BufferIOError) Error() string {
	return e.err.Error()
}

func (e *BufferIOError) Unwrap() error {
	return e.err
}

type BufferNoGeneratorError struct {
	err   error
	scope string
}

func NewBufferNoGeneratorError(scope string) *BufferNoGeneratorError {
	return &BufferNoGeneratorError{
		err:   errnie.Error(errors.New("generator not set")),
		scope: scope,
	}
}

func (e *BufferNoGeneratorError) Error() string {
	return e.err.Error()
}

func (e *BufferNoGeneratorError) Unwrap() error {
	return e.err
}

type BufferNoOutputError struct {
	err   error
	scope string
}

func NewBufferNoOutputError(scope string) *BufferNoOutputError {
	return &BufferNoOutputError{
		err:   errnie.Error(errors.New("output channel not set")),
		scope: scope,
	}
}

func (e *BufferNoOutputError) Error() string {
	return e.err.Error()
}

func (e *BufferNoOutputError) Unwrap() error {
	return e.err
}

type BufferNoInputError struct {
	err   error
	scope string
}

func NewBufferNoInputError(scope string) *BufferNoInputError {
	return &BufferNoInputError{
		err:   errnie.Error(errors.New("input channel not set")),
		scope: scope,
	}
}

func (e *BufferNoInputError) Error() string {
	return e.err.Error()
}

func (e *BufferNoInputError) Unwrap() error {
	return e.err
}

type BufferNoContextError struct {
	err   error
	scope string
}

func NewBufferNoContextError(scope string) *BufferNoContextError {
	return &BufferNoContextError{
		err:   errnie.Error(errors.New("context not set")),
		scope: scope,
	}
}

func (e *BufferNoContextError) Error() string {
	return e.err.Error()
}

func (e *BufferNoContextError) Unwrap() error {
	return e.err
}
