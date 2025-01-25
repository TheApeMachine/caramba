package provider

import "time"

/*
Event represents a generic event in the system that all providers must implement.
It provides a common interface for handling different types of events uniformly.
*/
type Event interface {
	ID() string
	Type() string
	Timestamp() time.Time
	Data() interface{}
	Metadata() map[string]interface{}
}

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
EventData represents the concrete implementation of event information
that providers can use as a base implementation.
*/
type EventData struct {
	Sequence    int64
	TeamID      string
	AgentID     string
	EventType   EventType
	Name        string
	Text        string
	PartialJSON string
	Error       error
	timestamp   time.Time
	metadata    map[string]interface{}
}

// NewEventData creates a new EventData instance with initialized fields
func NewEventData() *EventData {
	return &EventData{
		timestamp: time.Now(),
		metadata:  make(map[string]interface{}),
	}
}

// ID returns a unique identifier for the event
func (e *EventData) ID() string {
	return e.Name
}

// Type returns the string representation of the event type
func (e *EventData) Type() string {
	switch e.EventType {
	case EventStart:
		return "start"
	case EventChunk:
		return "chunk"
	case EventToolCall:
		return "tool_call"
	case EventError:
		return "error"
	case EventDone:
		return "done"
	default:
		return "unknown"
	}
}

// Timestamp returns when the event occurred
func (e *EventData) Timestamp() time.Time {
	return e.timestamp
}

// Data returns the event payload
func (e *EventData) Data() interface{} {
	return map[string]interface{}{
		"sequence":     e.Sequence,
		"team_id":      e.TeamID,
		"agent_id":     e.AgentID,
		"name":         e.Name,
		"text":         e.Text,
		"partial_json": e.PartialJSON,
		"error":        e.Error,
	}
}

// Metadata returns additional event context
func (e *EventData) Metadata() map[string]interface{} {
	return e.metadata
}
