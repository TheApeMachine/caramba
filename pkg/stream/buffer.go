package stream

import (
	"errors"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/event"
)

/*
Buffer provides a bidirectional streaming mechanism with encoding/decoding capabilities.
It connects a sender and receiver through pipes, handling data transformations using a pluggable codec system.
Buffer implements io.Reader, io.Writer, and io.Closer interfaces to support standard streaming operations.
*/
type Buffer struct {
	event  *event.Artifact
	fn     func(*event.Artifact) error
	Stream chan *event.Artifact
}

/*
NewBuffer creates a new Buffer with the specified receiver, sender, and handler function.
It sets up the necessary pipe connections and defaults to Gob encoding.

Parameters:
  - fn: A function that processes the event

Returns a configured Buffer instance that's ready to use.
*/
func NewBuffer(fn func(*event.Artifact) error) *Buffer {
	errnie.Debug("stream.NewBuffer")
	return &Buffer{
		event:  &event.Artifact{},
		fn:     fn,
		Stream: make(chan *event.Artifact, 64),
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

	// In streaming mode, we read from the stream
	if buffer.Stream != nil {
		artifact, ok := <-buffer.Stream
		if !ok {
			return 0, io.EOF
		}
		n, err = artifact.Read(p)
		if err != nil && err != io.EOF {
			errnie.Error(err)
			return 0, err
		}
		return n, err
	}

	// In non-streaming mode, we read from the event
	n, err = buffer.event.Read(p)
	if err != nil && err != io.EOF {
		errnie.Error(err)
		return 0, err
	}
	return n, err
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

	if len(p) == 0 {
		return 0, errnie.Error(errors.New("empty input"))
	}

	if buffer.event == nil {
		return 0, errnie.Error(errors.New("buffer event is nil"))
	}

	// Create a new event for this write operation
	newEvent := &event.Artifact{}

	// Log before Write
	id, _ := newEvent.Id()
	typ, _ := newEvent.Type()
	payload, _ := newEvent.Payload()
	errnie.Debug("Before event.Write", "event_id", id, "event_type", typ, "payload_length", len(payload))

	// Write to the new event
	if n, err = newEvent.Write(p); errnie.Error(err) != nil {
		return
	}

	// Log after Write
	id, _ = newEvent.Id()
	typ, _ = newEvent.Type()
	payload, _ = newEvent.Payload()
	errnie.Debug("After event.Write", "event_id", id, "event_type", typ, "payload_length", len(payload))

	// Process through handler function
	if err = buffer.fn(newEvent); errnie.Error(err) != nil {
		return
	}

	// Log after handler
	id, _ = newEvent.Id()
	typ, _ = newEvent.Type()
	payload, _ = newEvent.Payload()
	errnie.Debug("After handler", "event_id", id, "event_type", typ, "payload_length", len(payload))

	// Update the buffer's event
	buffer.event = newEvent

	// If we're in streaming mode, send the event to the stream
	if buffer.Stream != nil {
		select {
		case buffer.Stream <- newEvent:
			// Successfully sent to stream
		default:
			// Channel is full, log warning but don't block
			errnie.Debug("Warning: Stream channel is full, dropping event")
		}
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
	if buffer.Stream != nil {
		close(buffer.Stream)
	}
	return buffer.event.Close()
}
