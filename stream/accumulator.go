package stream

import (
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
	err    error
}

func NewAccumulator() *Accumulator {
	return &Accumulator{
		wg: &sync.WaitGroup{},
	}
}

func (accumulator *Accumulator) Clear() {
	accumulator.chunks = []*provider.Event{}
	accumulator.err = nil
}

func (accumulator *Accumulator) Generate(ctx context.Context, in <-chan *provider.Event) <-chan *provider.Event {
	out := make(chan *provider.Event)

	accumulator.wg.Add(1)

	go func() {
		defer close(out)
		defer accumulator.wg.Done()

		for event := range in {
			// Check for error events
			if event.Type == provider.EventError {
				accumulator.err = errors.New(event.Text)
				out <- event
				return
			}

			if event.Type == provider.EventChunk {
				accumulator.chunks = append(accumulator.chunks, event)
			}

			if event.Type == provider.EventStop {
				accumulator.chunks = append(accumulator.chunks, event)
			}

			out <- event
		}
	}()

	return out
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

	return strings.TrimSpace(out.String())
}

/*
Error returns any error that occurred during accumulation
*/
func (accumulator *Accumulator) Error() string {
	return errors.Unwrap(accumulator.err).Error()
}
