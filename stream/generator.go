package stream

import (
	"github.com/theapemachine/caramba/provider"
)

/*
Generator defines the interface for components that can generate and process
event streams. It provides a standardized way to handle provider Events,
combining real-time streaming capabilities with internal event accumulation.

Implementations of this interface should:
- Process input messages and generate event streams
- Support real-time event throughput for immediate feedback
- Maintain internal state for event accumulation
- Handle both structured and unstructured event data
- Provide access to accumulated results when needed
*/
type Generator interface {
	/*
		Generate processes a message and produces a stream of events.

		Parameters:
			message: The message to process and generate events from

		Returns:
			<-chan *provider.Event: A channel that streams the generated events
	*/
	Generate(*provider.Message) <-chan *provider.Event
}
