package message

import (
	"bytes"
	"io"

	"capnproto.org/go/capnp/v3"
	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/ai/toolcall"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type MessageOption func(Message) Message

func New(opts ...MessageOption) Message {
	errnie.Trace("message.New")

	var (
		arena = capnp.SingleSegment(nil)
		seg   *capnp.Segment
		msg   Message
		err   error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return errnie.Try(NewMessage(seg)).ToState(errnie.StateError)
	}

	if msg, err = NewRootMessage(seg); errnie.Error(err) != nil {
		return errnie.Try(NewMessage(seg)).ToState(errnie.StateError)
	}

	msg.SetUuid(uuid.New().String())
	msg.SetState(uint64(errnie.StatePending))

	for _, opt := range opts {
		opt(msg)
	}

	return datura.Register(msg)
}

func WithBytes(b []byte) MessageOption {
	return func(m Message) Message {
		errnie.Trace("message.WithBytes")
		errnie.Try(io.Copy(m, bytes.NewBuffer(b)))
		return m
	}
}

func WithMessage(msg *Message) MessageOption {
	return func(m Message) Message {
		errnie.Trace("message.WithMessage")
		m = errnie.Try(ReadRootMessage(m.Message()))
		return m
	}
}

func WithRole(role string) MessageOption {
	return func(m Message) Message {
		errnie.Trace("message.WithRole")
		m.SetRole(role)
		return m
	}
}

func WithContent(content string) MessageOption {
	return func(m Message) Message {
		errnie.Trace("message.WithContent")
		m.SetContent(content)
		return m
	}
}

func WithName(name string) MessageOption {
	return func(m Message) Message {
		errnie.Trace("message.WithName")
		m = errnie.Try(ReadRootMessage(m.Message()))
		m.SetName(name)
		return m
	}
}

func WithToolCalls(toolCalls []toolcall.ToolCallBuilder) MessageOption {
	return func(m Message) Message {
		errnie.Trace("message.WithToolCalls")

		// Create a new ToolCall_List with the same length as toolCalls
		tcList, err := toolcall.NewToolCall_List(m.Segment(), int32(len(toolCalls)))
		if errnie.Error(err) != nil {
			return errnie.Try(NewMessage(m.Segment())).ToState(errnie.StateError)
		}

		// Copy each ToolCall from the builders into the list
		for i, tc := range toolCalls {
			tcList.Set(i, *tc.ToolCall)
		}

		m.SetToolCalls(tcList)
		return m
	}
}
