package core

import (
	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/datura"
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

func (builder *ContextBuilder) AddMessage(artifact *datura.Artifact) {
	msg := NewMessageBuilder(WithArtifact(artifact))

	messages, err := builder.Messages()

	if errnie.Error(err) != nil {
		return
	}

	msgList, err := NewMessage_List(builder.Segment(), int32(messages.Len()+1))

	if errnie.Error(err) != nil {
		return
	}

	for i := range messages.Len() {
		msgList.Set(i, messages.At(i))
	}

	msgList.Set(messages.Len(), *msg.Message)
	builder.SetMessages(msgList)
}

func (ctx *ContextBuilder) Artifact() *datura.Artifact {
	message := ctx.Message()

	payload, err := message.Marshal()

	if errnie.Error(err) != nil {
		return nil
	}

	return datura.New(
		datura.WithPayload(payload),
		datura.WithRole(datura.ArtifactRoleAnswer),
		datura.WithScope(datura.ArtifactScopeContext),
	)
}

func (ctx *ContextBuilder) WithArtifact(artifact *datura.Artifact) *ContextBuilder {
	payload, err := artifact.DecryptPayload()

	if errnie.Error(err) != nil {
		return ctx
	}

	msgData, err := capnp.Unmarshal(payload)

	if errnie.Error(err) != nil {
		return ctx
	}

	msg, err := ReadRootContext(msgData)

	if errnie.Error(err) != nil {
		return ctx
	}

	ctx.Context = &msg
	return ctx
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
