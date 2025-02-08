package stream

import (
	"github.com/theapemachine/caramba/provider"
)

/*
Generator is an interface that all object must implement if they want to
become compatible with a stream of provider Events. In general, a Generator
is expected to have a throughput channel which is returned from the Generate
method, so the stream of events can be real-time for a direct user experience,
meanwhile the events should be accumulated internally, to be reconstructued into
a complete generated response, and used in any post-processing.
*/
type Generator interface {
	Generate(*provider.Message) <-chan *provider.Event
}
