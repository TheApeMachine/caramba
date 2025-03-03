package hub

import (
	"encoding/json"
	"fmt"
	"regexp"
)

// ANSI escape sequence regex patterns
var ansiRegex = regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)
var comprehensiveAnsiRegex = regexp.MustCompile(`(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]`)

// StripANSI removes ANSI escape sequences from the provided string
func StripANSI(s string) string {
	return comprehensiveAnsiRegex.ReplaceAllString(s, "")
}

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
	// Strip any ANSI escape codes from the message
	cleanMessage := StripANSI(message)

	return &Event{
		Origin:  origin,
		Topic:   topic,
		Type:    eventType,
		Role:    role,
		Message: cleanMessage,
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

func NewError(
	origin string, topic string, role string, message string,
) *Event {
	return NewEvent(
		origin, topic, role, EventTypeError, message, map[string]string{},
	)
}

func NewWarning(
	origin string, topic string, role string, message string,
) *Event {
	return NewEvent(
		origin, topic, role, EventTypeWarning, message, map[string]string{},
	)
}

func NewInfo(
	origin string, topic string, role string, message string,
) *Event {
	return NewEvent(
		origin, topic, role, EventTypeInfo, message, map[string]string{},
	)
}

func NewDebug(
	origin string, topic string, role string, message string,
) *Event {
	return NewEvent(
		origin, topic, role, EventTypeDebug, message, map[string]string{},
	)
}

func NewFatal(
	origin string, topic string, role string, message string,
) *Event {
	return NewEvent(
		origin, topic, role, EventTypeFatal, message, map[string]string{},
	)
}

func NewLog(
	origin string, topic string, role string, message string,
) *Event {
	return NewEvent(
		origin, topic, role, EventTypeLog, message, map[string]string{},
	)
}

func NewStatus(
	origin string, role string, message string,
) *Event {
	return NewEvent(
		origin, "ui", role, EventTypeStatus, message, map[string]string{},
	)
}

func NewMessage(
	origin string, message string,
) *Event {
	return NewEvent(
		origin, "ui", "agent", EventTypeMessage, message, map[string]string{},
	)
}

func NewChunk(
	origin string, message string,
) *Event {
	return NewEvent(
		origin, "ui", "agent", EventTypeChunk, message, map[string]string{},
	)
}

func NewToolCall(
	origin string, role string, message string,
) *Event {
	return NewEvent(
		origin, "metric", role, EventTypeToolCall, message, map[string]string{},
	)
}

func NewMetric(
	origin string, role string, message string,
) *Event {
	return NewEvent(
		origin, "metric", role, EventTypeMetric, message, map[string]string{},
	)
}

func NewClear(
	origin string, topic string, role string, message string,
) *Event {
	return NewEvent(
		origin, topic, role, EventTypeClear, message, map[string]string{},
	)
}

func SendTo(
	origin string, topic string, role string, message string,
) *Event {
	return NewEvent(
		origin, topic, role, EventTypeMessage, message, map[string]string{},
	)
}
