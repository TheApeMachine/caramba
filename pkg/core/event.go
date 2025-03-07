package core

import (
	"bytes"
	"encoding/json"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type EventData struct {
	Message   *Message    `json:"message"`
	ToolCalls []*ToolCall `json:"tool_calls"`
	Error     error       `json:"error"`
}

type Event struct {
	*EventData
	enc *json.Encoder
	dec *json.Decoder
	in  *bytes.Buffer
	out *bytes.Buffer
}

func NewEvent(
	message *Message,
	error error,
) *Event {
	errnie.Debug("NewEvent")

	in := bytes.NewBuffer([]byte{})
	out := bytes.NewBuffer([]byte{})

	event := &Event{
		EventData: &EventData{
			Message:   message,
			ToolCalls: make([]*ToolCall, 0),
			Error:     error,
		},
		enc: json.NewEncoder(out),
		dec: json.NewDecoder(in),
		in:  in,
		out: out,
	}

	// Pre-encode the event data to JSON for reading
	event.enc.Encode(event.EventData)

	return event
}

func (event *Event) Read(p []byte) (n int, err error) {
	errnie.Debug("Event.Read")

	if event.out.Len() == 0 {
		if err = errnie.NewErrIO(event.enc.Encode(event.EventData)); err != nil {
			return 0, err
		}
	}

	return event.out.Read(p)
}

func (event *Event) Write(p []byte) (n int, err error) {
	errnie.Debug("Event.Write", "p", string(p))

	// Reset the output buffer whenever we write new data
	if event.out.Len() > 0 {
		event.out.Reset()
	}

	// Write the incoming bytes to the input buffer
	n, err = event.in.Write(p)
	if err != nil {
		return n, err
	}

	// Try to decode the data from the input buffer
	// If it fails, we still return the bytes written but keep the error
	var buf EventData
	if decErr := event.dec.Decode(&buf); decErr == nil {
		// Only update if decoding was successful
		event.EventData.Message = buf.Message
		event.EventData.ToolCalls = buf.ToolCalls
		event.EventData.Error = buf.Error

		// Re-encode to the output buffer for subsequent reads
		if encErr := event.enc.Encode(event.EventData); encErr != nil {
			return n, errnie.NewErrIO(encErr)
		}
	}

	return n, nil
}

func (event *Event) Close() error {
	errnie.Debug("Event.Close")

	event.EventData.Message = nil
	event.EventData.ToolCalls = nil
	event.EventData.Error = nil
	return nil
}

func (event *Event) WithToolCalls(toolCalls ...*ToolCall) *Event {
	errnie.Debug("Event.WithToolCalls")

	event.EventData.ToolCalls = append(event.EventData.ToolCalls, toolCalls...)
	return event
}
