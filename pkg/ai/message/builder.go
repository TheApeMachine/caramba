package message

import (
	"bufio"
	"bytes"
	"io"

	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/ai/toolcall"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type MessageBuilder struct {
	Message *Message
	encoder *capnp.Encoder
	decoder *capnp.Decoder
	buffer  *bufio.ReadWriter
	State   MessageState
}

type MessageOption func(*MessageBuilder)

func New(opts ...MessageOption) *MessageBuilder {
	errnie.Trace("message.New")

	var (
		arena = capnp.SingleSegment(nil)
		seg   *capnp.Segment
		msg   Message
		err   error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil
	}

	if msg, err = NewRootMessage(seg); errnie.Error(err) != nil {
		return nil
	}

	shared := bytes.NewBuffer(nil)
	buffer := bufio.NewReadWriter(
		bufio.NewReader(shared),
		bufio.NewWriter(shared),
	)

	message := &MessageBuilder{
		Message: &msg,
		encoder: capnp.NewEncoder(buffer),
		decoder: capnp.NewDecoder(buffer),
		buffer:  buffer,
	}

	for _, opt := range opts {
		opt(message)
	}

	return message
}

func WithBytes(b []byte) MessageOption {
	return func(m *MessageBuilder) {
		errnie.Trace("message.WithBytes")

		if _, err := io.Copy(
			m.buffer, bytes.NewBuffer(b),
		); errnie.Error(err) != nil {
			return
		}
	}
}

func WithMessage(msg *Message) MessageOption {
	return func(m *MessageBuilder) {
		errnie.Trace("message.WithMessage")

		m.Message = msg
	}
}

func WithRole(role string) MessageOption {
	return func(msg *MessageBuilder) {
		errnie.Trace("message.WithRole")

		if err := msg.Message.SetRole(role); errnie.Error(err) != nil {
			return
		}
	}
}

func WithContent(content string) MessageOption {
	return func(msg *MessageBuilder) {
		errnie.Trace("message.WithContent")

		if err := msg.Message.SetContent(content); errnie.Error(err) != nil {
			return
		}
	}
}

func WithName(name string) MessageOption {
	return func(msg *MessageBuilder) {
		errnie.Trace("message.WithName")

		if err := msg.Message.SetName(name); errnie.Error(err) != nil {
			return
		}
	}
}

func WithToolCalls(toolCalls []toolcall.ToolCallBuilder) MessageOption {
	return func(msg *MessageBuilder) {
		errnie.Trace("message.WithToolCalls")

		// Create a new ToolCall_List with the same length as toolCalls
		tcList, err := toolcall.NewToolCall_List(msg.Message.Segment(), int32(len(toolCalls)))
		if errnie.Error(err) != nil {
			return
		}

		// Copy each ToolCall from the builders into the list
		for i, tc := range toolCalls {
			tcList.Set(i, *tc.ToolCall)
		}

		if err := msg.Message.SetToolCalls(tcList); errnie.Error(err) != nil {
			return
		}
	}
}
