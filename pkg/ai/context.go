package ai

import (
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/utils"
)

type ContextBuilder struct {
	*Context
}

type ContextOption func(*ContextBuilder)

func NewContextBuilder(opts ...ContextOption) *ContextBuilder {
	var (
		cpnp = utils.NewCapnp()
		ctx  Context
		err  error
	)

	if ctx, err = NewRootContext(cpnp.Seg); errnie.Error(err) != nil {
		return nil
	}

	builder := &ContextBuilder{
		Context: &ctx,
	}

	for _, opt := range opts {
		opt(builder)
	}

	return builder
}

func WithMessages(messages ...*MessageBuilder) ContextOption {
	return func(builder *ContextBuilder) {
		msgList, err := NewMessage_List(builder.Segment(), int32(len(messages)))
		if errnie.Error(err) != nil {
			return
		}

		for i, msg := range messages {
			msgList.Set(i, *msg.Message)
		}

		builder.SetMessages(msgList)
	}
}
