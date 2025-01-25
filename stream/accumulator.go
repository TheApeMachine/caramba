package stream

import (
	"context"
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
	chunks []provider.Event
	err    error
}

func NewAccumulator() *Accumulator {
	return &Accumulator{
		wg: &sync.WaitGroup{},
	}
}

func (accumulator *Accumulator) Generate(ctx context.Context, in <-chan provider.Event) <-chan provider.Event {
	out := make(chan provider.Event)

	accumulator.wg.Add(1)

	go func() {
		defer close(out)
		defer accumulator.wg.Done()

		for event := range in {
			// Check for error events
			if data, ok := event.Data().(map[string]interface{}); ok {
				if errVal, ok := data["error"]; ok {
					if err, ok := errVal.(error); ok {
						accumulator.err = err
						out <- event
						return
					}
				}
			}

			accumulator.chunks = append(accumulator.chunks, event)
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
	return accumulator.Compile().Data().(map[string]interface{})["text"].(string)
}

/*
Compile the chunks into a single provider Event, which can be used to evaluate
or process the completed response that was generated by the provider model.
*/
func (accumulator *Accumulator) Compile() provider.Event {
	// If there was an error, return an error event
	if accumulator.err != nil {
		errEvent := provider.NewEventData()
		errEvent.EventType = provider.EventError
		errEvent.Error = accumulator.err
		errEvent.Name = "accumulator_error"
		return errEvent
	}

	out := provider.NewEventData()
	out.EventType = provider.EventDone

	for _, chunk := range accumulator.chunks {
		if data, ok := chunk.Data().(map[string]interface{}); ok {
			if text, ok := data["text"].(string); ok {
				out.Text += text
			}
			if json, ok := data["partial_json"].(string); ok {
				out.PartialJSON += json
			}
		}
		if data, ok := chunk.Data().(map[string]interface{}); ok {
			if text, ok := data["text"].(string); ok {
				out.Text += text
			}
			if json, ok := data["partial_json"].(string); ok {
				out.PartialJSON += json
			}
		}
	}

	return out
}

/*
Error returns any error that occurred during accumulation
*/
func (accumulator *Accumulator) Error() error {
	return accumulator.err
}
