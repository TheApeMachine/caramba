package hub

import (
	"encoding/json"
	"fmt"
)

type EventType string

const (
	EventTypeUnknown  EventType = "unknown"
	EventTypeToolCall EventType = "tool_call"
	EventTypeChunk    EventType = "chunk"
	EventTypeMessage  EventType = "message"
	EventTypeError    EventType = "error"
	EventTypeWarning  EventType = "warning"
	EventTypeInfo     EventType = "info"
	EventTypeDebug    EventType = "debug"
	EventTypeFatal    EventType = "fatal"
	EventTypePanic    EventType = "panic"
	EventTypeLog      EventType = "log"
	EventTypeStatus   EventType = "status"
	EventTypeMetric   EventType = "metric"
	EventTypeClear    EventType = "clear"
)

type Event struct {
	ID      string
	Type    EventType
	Topic   string
	Origin  string
	Role    string
	Message string
	Meta    map[string]string
}

func NewEvent(
	origin string,
	topic string,
	role string,
	eventType EventType,
	message string,
	meta map[string]string,
) *Event {
	return &Event{
		Origin:  origin,
		Topic:   topic,
		Type:    eventType,
		Role:    role,
		Message: message,
		Meta:    meta,
	}
}

func (e *Event) String() string {
	return fmt.Sprintf("%s: %s", e.Type, e.Message)
}

func (e *Event) JSON() string {
	json, err := json.Marshal(e)
	if err != nil {
		return ""
	}
	return string(json)
}
