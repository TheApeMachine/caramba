package core

import (
	"bufio"
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
	dec    *json.Decoder
	enc    *json.Encoder
	buffer *bufio.ReadWriter
}

func NewEvent(
	message *Message,
	err error,
) *Event {
	errnie.Debug("core.NewEvent")

	buf := bytes.NewBuffer([]byte{})
	buffer := bufio.NewReadWriter(
		bufio.NewReader(buf),
		bufio.NewWriter(buf),
	)

	event := &Event{
		EventData: &EventData{
			Message:   message,
			ToolCalls: []*ToolCall{},
			Error:     err,
		},
		dec:    json.NewDecoder(buffer),
		enc:    json.NewEncoder(buffer),
		buffer: buffer,
	}

	event.enc.Encode(event.EventData)
	return event
}

// Removed the startStreaming() method entirely

func (event *Event) Read(p []byte) (n int, err error) {
	errnie.Debug("core.Event.Read")

	if err = event.buffer.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if n, err = event.buffer.Read(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	errnie.Debug("core.Event.Read", "n", n, "err", err)

	return n, err
}

// Synchronous Write, no channels or goroutines
func (event *Event) Write(p []byte) (n int, err error) {
	errnie.Debug("core.Event.Write", "p", string(p))

	if n, err = event.buffer.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	msg := &Message{}

	if err = event.dec.Decode(msg); err != nil {
		errnie.NewErrIO(err)
		return
	}

	event.Message = msg

	if err = event.enc.Encode(event.EventData); err != nil {
		errnie.NewErrIO(err)
		return
	}

	return len(p), err
}

func (event *Event) Close() error {
	errnie.Debug("core.Event.Close")

	event.buffer.Flush()
	event.EventData = nil
	event.dec = nil
	event.enc = nil

	return nil
}

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
