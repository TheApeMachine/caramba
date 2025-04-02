package ai

import (
	"errors"

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

func (builder *ContextBuilder) Validate(scope string) error {
	if builder.Context == nil {
		return NewAgentContextValidationError(scope, errors.New("context not set"))
	}

	msgList, err := builder.Messages()

	if errnie.Error(err) != nil {
		return NewAgentContextValidationError(scope, err)
	}

	for i := range msgList.Len() {
		msg := msgList.At(i)
		role, err := msg.Role()

		if errnie.Error(err) != nil {
			return NewAgentContextValidationError(scope, err)
		}

		if role != "system" && role != "user" && role != "assistant" {
			return NewAgentContextValidationError(
				scope, errors.New("first message not a system message"),
			)
		}
	}

	return nil
}
