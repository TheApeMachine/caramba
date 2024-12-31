package stream

import "github.com/theapemachine/caramba/provider"

/*
Accumulator is a wrapper for a provider Event stream which has both
a direct throughput, and an accumulating buffer. This allows you to
take advantage of both real-time feedback flow, like output to a
user, or frontend of some kind, while also being able to use the
full output once the provider model is finished generating a response.
*/
type Accumulator struct {
	chunks []provider.Event
}

func NewAccumulator() *Accumulator {
	return &Accumulator{}
}

func (accumulator *Accumulator) Generate(in <-chan provider.Event) <-chan provider.Event {
	out := make(chan provider.Event)

	go func() {
		for event := range in {
			accumulator.chunks = append(accumulator.chunks, event)
			out <- event
		}
	}()

	return out
}
