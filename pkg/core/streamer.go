package core

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

/*
Streamer implements io.ReadWriteCloser for a Generator.

It allows us to have super flexible message passing between any components,
and when things get more complex, there is a rich ecosystem within the
standard io package to help us out, like TeeReader, TeeWriter, MultiReader,
MultiWriter, Pipe, etc.
*/
type Streamer struct {
	buffer *stream.Buffer
	topics []string
}

type StreamerOption func(*Streamer)

/*
NewStreamer creates a Streaner which wraps the Generator that is passed in.

We can just use system.NewHub() to set the hub, since it is a sync.Once,
so it always returns the same hub.
*/
func NewStreamer(opts ...StreamerOption) *Streamer {
	errnie.Debug("core.NewStreamer")

	streamer := &Streamer{
		topics: []string{},
	}

	for _, opt := range opts {
		opt(streamer)
	}

	return streamer
}

func (streamer *Streamer) ID() string {
	return streamer.buffer.ID()
}

/*
Read implements io.Reader for the Streamer.
*/
func (streamer *Streamer) Read(p []byte) (n int, err error) {
	errnie.Debug("core.Streamer.Read", "id", streamer.ID(), "bytes", n)

	if err := streamer.Validate("core.Streamer.Read"); err != nil {
		return 0, err
	}

	if n, err = streamer.buffer.Read(p); err != nil {
		if err == io.EOF {
			return n, io.EOF
		}
		return 0, NewStreamerIOError("core.Streamer.Read", err)
	}

	if n == 0 {
		return 0, io.EOF
	}

	return
}

/*
Write implements io.Writer for the Streamer.
*/
func (streamer *Streamer) Write(p []byte) (n int, err error) {
	errnie.Debug("core.Streamer.Write", "id", streamer.ID())

	if err := streamer.Validate("core.Streamer.Write"); err != nil {
		return 0, err
	}

	if n, err = streamer.buffer.Write(p); err != nil {
		return 0, NewStreamerIOError("core.Streamer.Write", err)
	}

	return
}

/*
Close implements io.Closer for the Streamer.
*/
func (streamer *Streamer) Close() (err error) {
	errnie.Debug("core.Streamer.Close", "id", streamer.ID())

	if err = streamer.Validate("core.Streamer.Close"); err != nil {
		return err
	}

	if err = streamer.buffer.Close(); err != nil {
		return NewStreamerIOError("core.Streamer.Close", err)
	}

	return
}

func WithGenerator(generator stream.Generator) StreamerOption {
	return func(streamer *Streamer) {
		streamer.buffer = stream.NewBuffer(
			stream.WithGenerator(generator),
		)
	}
}

func WithTopics(topics ...string) StreamerOption {
	return func(streamer *Streamer) {
		streamer.topics = topics
	}
}

// WithContext sets the context for the streamer and its underlying buffer
func WithContext(ctx context.Context) StreamerOption {
	return func(streamer *Streamer) {
		if streamer.buffer != nil {
			stream.WithCancel(ctx, nil)(streamer.buffer)
		}
	}
}

/*
Validate implements the StreamerValidator interface for the Streamer.
*/
func (streamer *Streamer) Validate(scope string) error {
	if streamer.buffer == nil {
		return NewStreamerNoBufferError(scope)
	}

	return nil
}

type StreamerIOError struct {
	err   error
	scope string
}

func NewStreamerIOError(scope string, err error) *StreamerIOError {
	if err == nil || err == io.EOF {
		return &StreamerIOError{
			err:   err,
			scope: scope,
		}
	}

	return &StreamerIOError{
		err:   errnie.Error(fmt.Errorf("%s - %w", scope, err)),
		scope: scope,
	}
}

func (e *StreamerIOError) Error() string {
	return e.err.Error()
}

func (e *StreamerIOError) Unwrap() error {
	if e == nil || e.err == nil {
		return nil
	}
	return e.err
}

type StreamerNoHubError struct {
	err   error
	scope string
}

func NewStreamerNoHubError(scope string) *StreamerNoHubError {
	return &StreamerNoHubError{
		err:   errnie.Error(fmt.Errorf("%s - %w", scope, errors.New("hub not set"))),
		scope: scope,
	}
}

func (e *StreamerNoHubError) Error() string {
	return e.err.Error()
}

func (e *StreamerNoHubError) Unwrap() error {
	return e.err
}

type StreamerNoBufferError struct {
	err   error
	scope string
}

func NewStreamerNoBufferError(scope string) *StreamerNoBufferError {
	return &StreamerNoBufferError{
		err:   errnie.Error(fmt.Errorf("%s - %w", scope, errors.New("buffer not set"))),
		scope: scope,
	}
}

func (e *StreamerNoBufferError) Error() string {
	return e.err.Error()
}

func (e *StreamerNoBufferError) Unwrap() error {
	return e.err
}

func (streamer *Streamer) Topics() []string {
	return streamer.topics
}
