/*
Package core provides the central types and functionality for the application.
*/
package core

import (
	"bytes"
	"encoding/gob"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

/*
EventData holds the content data for an Event.
It contains a message, tool calls, and any error information.
*/
type EventData struct {
	Message   *MessageData `json:"message"`
	ToolCalls []*ToolCall  `json:"tool_calls"`
	Error     string       `json:"error"`
}

/*
Event represents a communication unit in the system.
It wraps EventData with a Buffer for streaming capabilities.
*/
type Event struct {
	*EventData
	*stream.Buffer `json:"-" gob:"-"` // Exclude from serialization
}

/*
NewEvent creates and initializes a new Event with the given message and error.
It sets up the internal buffer and pre-buffers the EventData if a message is provided.
*/
func NewEvent(
	message *Message,
	err error,
) *Event {
	errnie.Debug("core.NewEvent")

	errMsg := ""

	if err != nil {
		errMsg = err.Error()
	}

	msg := &MessageData{}

	if message != nil {
		msg.Role = message.Role
		msg.Name = message.Name
		msg.Content = message.Content
	}

	event := &Event{
		EventData: &EventData{
			Message:   msg,
			ToolCalls: []*ToolCall{},
			Error:     errMsg,
		},
	}

	event.Buffer = stream.NewBuffer(
		&EventData{},
		event.EventData,
		func(msg any) error {
			if decoded, ok := msg.(*EventData); ok {
				event.EventData = decoded
			}
			return nil
		},
	)

	if message != nil {
		buf := bytes.NewBuffer([]byte{})
		// Only encode the EventData, not the full Event with Buffer
		ed := &EventData{
			Message:   message.MessageData,
			ToolCalls: event.EventData.ToolCalls,
			Error:     event.EventData.Error,
		}
		gob.NewEncoder(buf).Encode(ed)
		event.Write(buf.Bytes())
		errnie.Debug("core.NewEvent", "prebuffered", event.String())
	}

	return event
}

/*
String returns the content of the Event's message as a string.
*/
func (event *Event) String() string {
	errnie.Debug("core.Event.String")
	return event.Message.Content
}

/*
Read implements the io.Reader interface for Event.
It delegates to the underlying Buffer's Read method.
*/
func (event *Event) Read(p []byte) (n int, err error) {
	errnie.Debug("core.Event.Read")
	return event.Buffer.Read(p)
}

/*
Write implements the io.Writer interface for Event.
It performs synchronous writes with no channels or goroutines.
*/
func (event *Event) Write(p []byte) (n int, err error) {
	errnie.Debug("core.Event.Write", "p", string(p))
	return event.Buffer.Write(p)
}

/*
Close implements the io.Closer interface for Event.
It delegates to the underlying Buffer's Close method.
*/
func (event *Event) Close() error {
	errnie.Debug("core.Event.Close")
	return event.Buffer.Close()
}

/*
WithToolCalls adds the provided tool calls to the Event.
It initializes EventData if needed and returns the updated Event.
*/
func (event *Event) WithToolCalls(toolCalls ...*ToolCall) *Event {
	errnie.Debug("core.Event.WithToolCalls")

	if event.EventData == nil {
		event.EventData = &EventData{
			ToolCalls: []*ToolCall{},
		}
	}

	event.EventData.ToolCalls = append(event.EventData.ToolCalls, toolCalls...)
	return event
}
