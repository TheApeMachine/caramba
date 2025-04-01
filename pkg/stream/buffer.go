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
		ctx:    ctx,
		cancel: cancel,
		in:     make(chan *datura.Artifact),
		out:    make(chan *datura.Artifact),
	}

	for _, opt := range opts {
		opt(buffer)
	}

	return buffer
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
	errnie.Debug("stream.Buffer.Read")

	if buffer.out == nil {
		return 0, io.EOF
	}

	select {
	case <-buffer.ctx.Done():
		errnie.Debug("stream.Buffer.Read", "ctx.Done()")
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
	errnie.Debug("stream.Buffer.Write")

	if len(p) == 0 {
		return 0, errnie.Error(errors.New("empty input"))
	}

	buffer.in <- datura.Unmarshal(p)

	return len(p), nil
}

/*
Close implements the io.Closer interface.
It properly closes both the pipe reader and writer to prevent resource leaks.

Returns any error encountered during the closing process.
*/
func (buffer *Buffer) Close() error {
	errnie.Debug("stream.Buffer.Close")
	buffer.cancel()
	return nil
}

func WithCancel(ctx context.Context, cancel context.CancelFunc) BufferOption {
	return func(buffer *Buffer) {
		buffer.ctx = ctx
		buffer.cancel = cancel
	}
}

func WithGenerator(generator Generator) BufferOption {
	return func(buffer *Buffer) {
		buffer.generator = generator
		buffer.out = generator.Generate(buffer.in)
	}
}
