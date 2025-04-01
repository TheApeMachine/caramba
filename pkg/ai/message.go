package ai

import (
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/utils"
)

type MessageBuilder struct {
	*Message
}

type MessageOption func(*MessageBuilder)

func NewMessageBuilder(opts ...MessageOption) *MessageBuilder {
	var (
		cpnp = utils.NewCapnp()
		msg  Message
		err  error
	)

	if msg, err = NewRootMessage(cpnp.Seg); errnie.Error(err) != nil {
		return nil
	}

	builder := &MessageBuilder{
		Message: &msg,
	}

	for _, opt := range opts {
		opt(builder)
	}

	return builder
}

func WithRole(role string) MessageOption {
	return func(builder *MessageBuilder) {
		builder.SetRole(role)
	}
}

func WithName(name string) MessageOption {
	return func(builder *MessageBuilder) {
		builder.SetName(name)
	}
}

func WithContent(content string) MessageOption {
	return func(builder *MessageBuilder) {
		builder.SetContent(content)
	}
}
