package stream

import (
	"context"
	"errors"
	"io"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Buffer provides a bidirectional streaming mechanism with encoding/decoding capabilities.
It connects a sender and receiver through pipes, handling data transformations using a pluggable codec system.
Buffer implements io.Reader, io.Writer, and io.Closer interfaces to support standard streaming operations.
*/
type Buffer struct {
	pctx      context.Context
	ctx       context.Context
	cancel    context.CancelFunc
	in        chan *datura.Artifact
	out       chan *datura.Artifact
	generator Generator
}

type BufferOption func(*Buffer)

/*
NewBuffer creates a new Buffer with the specified receiver, sender, and handler function.
It sets up the necessary pipe connections and defaults to Gob encoding.

Parameters:
  - fn: A function that processes the event

Returns a configured Buffer instance that's ready to use.
*/
func NewBuffer(opts ...BufferOption) *Buffer {
	errnie.Debug("stream.NewBuffer")

	ctx, cancel := context.WithCancel(context.Background())

	buffer := &Buffer{
		pctx:   ctx,
		ctx:    ctx,
		cancel: cancel,
		in:     make(chan *datura.Artifact, 64),
		out:    make(chan *datura.Artifact, 64),
	}

	for _, opt := range opts {
		opt(buffer)
	}

	return buffer
}

func (buffer *Buffer) ID() string {
	return buffer.generator.ID()
}

/*
Read implements the io.Reader interface.
It reads data from the pipe reader, which contains data encoded by the Write method.

Parameters:
  - p: Byte slice where read data will be stored

Returns:
  - n: Number of bytes read
  - err: Any error encountered during reading
*/
func (buffer *Buffer) Read(p []byte) (n int, err error) {
	if err := buffer.Validate("stream.Buffer.Read"); err != nil {
		return 0, err
	}

	if buffer.out == nil {
		return 0, NewBufferIOError(
			"stream.Buffer.Read",
			errors.New("output channel not set"),
		)
	}

	select {
	case <-buffer.pctx.Done():
		buffer.Close()
		return 0, io.EOF
	case <-buffer.ctx.Done():
		buffer.Close()
		return 0, io.EOF
	case artifact := <-buffer.out:
		return artifact.Read(p)
	}
}

/*
Write implements the io.Writer interface.
It decodes incoming data into the receiver, applies the handler function,
then asynchronously encodes the sender's data back into the pipe.

Parameters:
  - p: Byte slice containing data to be written

Returns:
  - n: Number of bytes written
  - err: Any error encountered during writing
*/
func (buffer *Buffer) Write(p []byte) (n int, err error) {
	if err := buffer.Validate("stream.Buffer.Write"); err != nil {
		return 0, err
	}

	if len(p) == 0 {
		return 0, NewBufferIOError("stream.Buffer.Write", errors.New("empty input"))
	}

	artifact := datura.Unmarshal(p)
	buffer.in <- artifact

	return len(p), nil
}

/*
Close implements the io.Closer interface.
It properly closes both the pipe reader and writer to prevent resource leaks.

Returns any error encountered during the closing process.
*/
func (buffer *Buffer) Close() error {
	if err := buffer.Validate("stream.Buffer.Close"); err != nil {
		return err
	}

	buffer.cancel()

	close(buffer.in)
	close(buffer.out)

	return nil
}

func WithCancel(ctx context.Context, cancel context.CancelFunc) BufferOption {
	return func(buffer *Buffer) {
		buffer.pctx = ctx
	}
}

func WithGenerator(generator Generator) BufferOption {
	return func(buffer *Buffer) {
		buffer.generator = generator
		buffer.out = generator.Generate(buffer.in)
	}
}

func (buffer *Buffer) Validate(scope string) error {
	if buffer.generator == nil {
		return NewBufferNoGeneratorError(scope)
	}

	if buffer.out == nil {
		return NewBufferNoOutputError(scope)
	}

	if buffer.in == nil {
		return NewBufferNoInputError(scope)
	}

	if buffer.pctx == nil {
		return NewBufferNoContextError(scope)
	}

	return nil
}
