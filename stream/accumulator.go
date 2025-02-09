package stream

import (
	"bufio"
	"context"
	"errors"
	"strings"
	"sync"

	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/errnie"
)

/*
Accumulator is a wrapper for a provider Event stream which has both
a direct throughput and an accumulating buffer. This allows you to
take advantage of both real-time feedback flow, like output to a
user or frontend of some kind, while also being able to use the
full output once the provider model is finished generating a response.
The Accumulator implements io.ReadWriteCloser for flexible I/O operations.
*/
type Accumulator struct {
	wg     *sync.WaitGroup    // WaitGroup for synchronization
	chunks []*provider.Event  // Collected event chunks
	buffer *bufio.ReadWriter  // Buffer for I/O operations
	after  []func(str string) // Functions to run after accumulation
	err    error              // Last error encountered
}

/*
NewAccumulator creates and returns a new Accumulator instance with initialized
synchronization primitives and I/O buffer.

Returns:

	*Accumulator: A new Accumulator instance ready for use
*/
func NewAccumulator() *Accumulator {
	return &Accumulator{
		wg:     &sync.WaitGroup{},
		buffer: bufio.NewReadWriter(bufio.NewReader(nil), bufio.NewWriter(nil)),
	}
}

/*
Clear resets the accumulator's state by clearing the event chunks and error state.
This method should be called before starting a new accumulation session.
*/
func (accumulator *Accumulator) Clear() {
	errnie.Debug("clearing accumulator", "chunks", len(accumulator.chunks))
	accumulator.chunks = []*provider.Event{}
	accumulator.err = nil
}

/*
After registers one or more functions to be executed after the accumulation
is complete. These functions receive the accumulated text as their argument.

Parameters:

	fns: Variable number of functions to execute after accumulation
*/
func (accumulator *Accumulator) After(fns ...func(str string)) {
	errnie.Debug("registering after function", "fns", len(fns))

	accumulator.after = append(accumulator.after, fns...)
}

/*
Generate processes an input event stream and produces an output event stream
while accumulating the events. It manages the accumulator's state and executes
registered after-functions upon completion.

Parameters:

	in: Input channel of provider Events to process

Returns:

	<-chan *provider.Event: Output channel of processed events
*/
func (accumulator *Accumulator) Generate(in <-chan *provider.Event) <-chan *provider.Event {
	_, cancel := context.WithCancel(context.TODO())
	out := make(chan *provider.Event)

	accumulator.wg.Add(1)

	go func() {
		defer close(out)
		defer accumulator.wg.Done()
		defer cancel()

		accumulator.Clear()

		for event := range in {
			if event.Type == provider.EventError {
				accumulator.err = errors.New(event.Text)
			}

			out <- event
			accumulator.chunks = append(accumulator.chunks, event)
		}

		for _, fn := range accumulator.after {
			fn(accumulator.String())
		}
	}()

	return out
}

/*
Append adds a new text chunk to the accumulator as an event.

Parameters:

	str: The text to append as a new event chunk
*/
func (accumulator *Accumulator) Append(str string) {
	if str == "" {
		return
	}

	accumulator.chunks = append(accumulator.chunks, &provider.Event{
		Type: provider.EventChunk,
		Text: str,
	})
}

/*
Wait blocks until the accumulator has finished processing all events.
This is useful when you need to ensure all processing is complete at
the end of a pipeline.
*/
func (accumulator *Accumulator) Wait() {
	accumulator.wg.Wait()
}

/*
String returns the accumulated text as a single string, with proper
whitespace handling and newline normalization.

Returns:

	string: The complete accumulated text
*/
func (accumulator *Accumulator) String() string {
	var out strings.Builder

	for _, chunk := range accumulator.chunks {
		if chunk.Type == provider.EventChunk && chunk.Text != "" {
			out.WriteString(chunk.Text)
		}
	}

	return out.String()
}

/*
Error returns any error that occurred during accumulation as a string.

Returns:

	string: The error message, or empty string if no error occurred
*/
func (accumulator *Accumulator) Error() string {
	return errors.Unwrap(accumulator.err).Error()
}

/*
Read implements io.Reader. It reads data from the internal buffer into
the provided byte slice.

Parameters:

	p: Byte slice to read data into

Returns:

	n: Number of bytes read
	err: Any error that occurred during reading
*/
func (accumulator *Accumulator) Read(p []byte) (n int, err error) {
	return accumulator.buffer.Read(p)
}

/*
Write implements io.Writer. It writes data from the provided byte slice
into the internal buffer.

Parameters:

	p: Byte slice containing data to write

Returns:

	n: Number of bytes written
	err: Any error that occurred during writing
*/
func (accumulator *Accumulator) Write(p []byte) (n int, err error) {
	return accumulator.buffer.Write(p)
}

/*
Close implements io.Closer. It flushes any buffered data and releases
resources.

Returns:

	error: Any error that occurred during closing
*/
func (accumulator *Accumulator) Close() error {
	return accumulator.buffer.Flush()
}
