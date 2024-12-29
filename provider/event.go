package provider

/*
EventType is an enum of the different types of events that can occur.
*/
type EventType uint

const (
	EventStart EventType = iota
	EventChunk
	EventToolCall
	EventError
	EventDone
)

/*
Event is the generic response type that we convert all provider responses
into, so we can handle them uniformly.
*/
type Event struct {
	Sequence    int64
	TeamID      string
	AgentID     string
	Type        EventType
	Text        string
	PartialJSON string
	Error       error
}
