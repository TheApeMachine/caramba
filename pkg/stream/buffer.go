package stream

import (
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
	artifact *datura.Artifact
	fn       func(*datura.Artifact) error
}

/*
NewBuffer creates a new Buffer with the specified receiver, sender, and handler function.
It sets up the necessary pipe connections and defaults to Gob encoding.

Parameters:
  - fn: A function that processes the event

Returns a configured Buffer instance that's ready to use.
*/
func NewBuffer(fn func(*datura.Artifact) error) *Buffer {
	errnie.Debug("stream.NewBuffer")
	return &Buffer{
		artifact: datura.New(),
		fn:       fn,
	}
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

	if buffer.artifact == nil {
		return 0, io.EOF
	}

	n, err = buffer.artifact.Read(p)

	if err != nil {
		if err == io.EOF {
			return n, err
		}

		return n, errnie.Error(err)
	}

	if n == 0 {
		return 0, io.EOF
	}

	return n, errnie.Error(err)
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

	if buffer.artifact == nil {
		return 0, errnie.Error(errors.New("buffer artifact is nil"))
	}

	if n, err = buffer.artifact.Write(p); errnie.Error(err) != nil {
		return
	}

	if err = buffer.fn(buffer.artifact); errnie.Error(err) != nil {
		return
	}

	return n, nil
}

/*
Close implements the io.Closer interface.
It properly closes both the pipe reader and writer to prevent resource leaks.

Returns any error encountered during the closing process.
*/
func (buffer *Buffer) Close() error {
	errnie.Debug("stream.Buffer.Close")
	return buffer.artifact.Close()
}
