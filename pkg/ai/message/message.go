package message

import (
	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/ai/toolcall"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type MessageBuilder struct {
	Message *Message
}

type MessageOption func(*MessageBuilder)

func NewMessageBuilder(opts ...MessageOption) *MessageBuilder {
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

	message := &MessageBuilder{
		Message: &msg,
	}

	for _, opt := range opts {
		opt(message)
	}

	return message
}

func WithRole(role string) MessageOption {
	return func(msg *MessageBuilder) {
		if err := msg.Message.SetRole(role); errnie.Error(err) != nil {
			return
		}
	}
}

func WithContent(content string) MessageOption {
	return func(msg *MessageBuilder) {
		if err := msg.Message.SetContent(content); errnie.Error(err) != nil {
			return
		}
	}
}

func WithName(name string) MessageOption {
	return func(msg *MessageBuilder) {
		if err := msg.Message.SetName(name); errnie.Error(err) != nil {
			return
		}
	}
}

func WithToolCalls(toolCalls []toolcall.ToolCallBuilder) MessageOption {
	return func(msg *MessageBuilder) {
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
