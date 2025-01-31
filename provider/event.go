package provider

import (
	"time"

	"github.com/goccy/go-json"
	"github.com/google/uuid"
)

/*
EventType is an enum of the different types of events that can occur.
*/
type EventType uint

const (
	EventStart EventType = iota
	EventChunk
	EventFunction
	EventError
	EventStop
)

/*
Event represents the concrete implementation of event information
that providers can use as a base implementation.
*/
type Event struct {
	ID          uuid.UUID      `json:"id"`
	Sequence    int64          `json:"sequence"`
	Origin      string         `json:"origin"`
	Type        EventType      `json:"type"`
	Text        string         `json:"text"`
	PartialJSON string         `json:"partial_json"`
	Metadata    map[string]any `json:"metadata"`
}

// NewEvent creates a new Event instance with initialized fields
func NewEvent(
	origin string,
	eventType EventType,
	text string,
	partialJSON string,
	metadata map[string]any,
) *Event {
	return &Event{
		ID:          uuid.New(),
		Sequence:    time.Now().UTC().UnixNano(),
		Origin:      origin,
		Type:        eventType,
		Text:        text,
		PartialJSON: partialJSON,
		Metadata:    metadata,
	}
}

func (event *Event) Marshal() []byte {
	json, err := json.Marshal(event)
	if err != nil {
		return nil
	}
	return json
}
