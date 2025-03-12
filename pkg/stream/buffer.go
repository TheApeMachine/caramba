package stream

import (
	"context"
	"io"
	"time"

	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Buffer provides a bidirectional streaming mechanism with encoding/decoding capabilities.
It connects a sender and receiver through pipes, handling data transformations using a pluggable codec system.
Buffer implements io.Reader, io.Writer, and io.Closer interfaces to support standard streaming operations.
*/
type Buffer struct {
	receiver io.Writer
	sender   io.Reader
	handler  func(any) error
	codec    Codec
	pr       *io.PipeReader
	pw       *io.PipeWriter
}

/*
NewBuffer creates a new Buffer with the specified receiver, sender, and handler function.
It sets up the necessary pipe connections and defaults to Gob encoding.

Parameters:
  - receiver: The destination where decoded data will be written
  - sender: The source from which data will be read and encoded
  - handler: A function that processes the receiver after data is decoded

Returns a configured Buffer instance that's ready to use.
*/
func NewBuffer(
	receiver io.Writer,
	sender io.Reader,
	handler func(any) error,
) *Buffer {
	errnie.Debug("stream.NewBuffer")

	pr, pw := io.Pipe()

	buf := &Buffer{
		receiver: receiver,
		sender:   sender,
		handler:  handler,
		pr:       pr,
		pw:       pw,
	}

	// Default to Gob encoding, can be overridden by calling
	// WithCodec with a different codec.
	return buf.WithCodec(NewCodec(&GobCodec{}))
}

/*
WithCodec attaches a specific codec implementation to the Buffer.
This allows customizing how data is encoded and decoded during streaming operations.

Parameters:
  - codec: The Codec implementation to use for data transformation

Returns the Buffer instance with the new codec configured.
*/
func (buffer *Buffer) WithCodec(codec Codec) *Buffer {
	errnie.Debug("stream.Buffer.WithCodec")

	buffer.codec = codec
	buffer.codec.WithPipes(buffer.pr, buffer.pw)
	return buffer
}

/*
Stream continuously sends data from the sender to the given channel in a non-blocking manner.
It runs in a separate goroutine and respects context cancellation.

Parameters:
  - ctx: Context for cancellation control
  - receiver: Channel where encoded data will be sent
*/
func (buffer *Buffer) Stream(ctx context.Context, receiver chan any) {
	errnie.Debug("stream.Buffer.Stream")

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case receiver <- buffer.receiver:
				buffer.codec.Encode(buffer.sender)
			default:
				time.Sleep(100 * time.Millisecond)
			}
		}
	}()
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
	return buffer.pr.Read(p)
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
	errnie.Debug("stream.Buffer.Write", "p", string(p))

	if err = buffer.codec.Decode(buffer.receiver); err != nil {
		errnie.NewErrIO(err)
		return 0, err
	}

	if err = buffer.handler(buffer.receiver); err != nil {
		errnie.NewErrIO(err)
		return 0, err
	}

	go func() {
		defer buffer.pw.Close()

		if err = buffer.codec.Encode(buffer.sender); err != nil {
			errnie.NewErrIO(err)
			buffer.pr.CloseWithError(err)
			buffer.pw.CloseWithError(err)
		}
	}()

	return len(p), nil
}

/*
Close implements the io.Closer interface.
It properly closes both the pipe reader and writer to prevent resource leaks.

Returns any error encountered during the closing process.
*/
func (buffer *Buffer) Close() error {
	errnie.Debug("stream.Buffer.Close")

	buffer.pr.Close()
	buffer.pw.Close()
	return nil
}
