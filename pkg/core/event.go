/*
Package core provides the central types and functionality for the application.
*/
package core

import (
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

/*
EventData holds the content data for an Event.
It contains a message, tool calls, and any error information.
*/
type EventData struct {
	Message   *Message    `json:"message"`
	ToolCalls []*ToolCall `json:"tool_calls"`
	Error     error       `json:"error"`
}

/*
Event represents a communication unit in the system.
It wraps EventData with a Buffer for streaming capabilities.
*/
type Event struct {
	*EventData
	*stream.Buffer
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

	event := &Event{
		EventData: &EventData{
			Message:   message,
			ToolCalls: []*ToolCall{},
			Error:     err,
		},
	}

	event.Buffer = stream.NewBuffer(
		event,
		event,
		func(evt any) error {
			event.EventData = evt.(*Event).EventData
			return nil
		},
	)

	if message != nil {
		// Pre-buffer the EventData, since we have the values.
		if _, err := io.Copy(event, message); err != nil {
			errnie.NewErrIO(err)
		}

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
