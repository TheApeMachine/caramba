package stream

import (
	"bufio"
	"context"
	"errors"
	"strings"
	"sync"

	"github.com/theapemachine/caramba/provider"
)

/*
Accumulator is a wrapper for a provider Event stream which has both
a direct throughput, and an accumulating buffer. This allows you to
take advantage of both real-time feedback flow, like output to a
user, or frontend of some kind, while also being able to use the
full output once the provider model is finished generating a response.
*/
type Accumulator struct {
	wg     *sync.WaitGroup
	chunks []*provider.Event
	buffer *bufio.ReadWriter
	after  []func(str string)
	err    error
}

func NewAccumulator() *Accumulator {
	return &Accumulator{
		wg:     &sync.WaitGroup{},
		buffer: bufio.NewReadWriter(bufio.NewReader(nil), bufio.NewWriter(nil)),
	}
}

func (accumulator *Accumulator) Clear() {
	accumulator.chunks = []*provider.Event{}
	accumulator.err = nil
}

func (accumulator *Accumulator) After(fns ...func(str string)) {
	accumulator.after = append(accumulator.after, fns...)
}

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

func (accumulator *Accumulator) Append(str string) {
	accumulator.chunks = append(accumulator.chunks, &provider.Event{
		Type: provider.EventChunk,
		Text: str,
	})
}

/*
Wait provides a method to wait for the accumulator to finish processing.
Useful if you are at the end of a pipeline and there is no out channel to
for throughputting the final result.
*/
func (accumulator *Accumulator) Wait() {
	accumulator.wg.Wait()
}

/*
String returns the accumulated text as a string
*/
func (accumulator *Accumulator) String() string {
	var out strings.Builder

	for _, chunk := range accumulator.chunks {
		if chunk.Type == provider.EventChunk {
			out.WriteString(chunk.Text)
		}
	}

	buf := strings.TrimSpace(out.String())
	if !strings.HasSuffix(buf, "\n") {
		buf += "\n"
	}

	return strings.TrimSpace(out.String())
}

/*
Error returns any error that occurred during accumulation
*/
func (accumulator *Accumulator) Error() string {
	return errors.Unwrap(accumulator.err).Error()
}

/*
Read ...
*/
func (accumulator *Accumulator) Read(p []byte) (n int, err error) {
	return accumulator.buffer.Read(p)
}

/*
Write ...
*/
func (accumulator *Accumulator) Write(p []byte) (n int, err error) {
	return accumulator.buffer.Write(p)
}

/*
Close ...
*/
func (accumulator *Accumulator) Close() error {
	return accumulator.buffer.Flush()
}
