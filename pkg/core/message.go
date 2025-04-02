package core

import (
	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/datura"
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

func (builder *MessageBuilder) Artifact() *datura.Artifact {
	return datura.New(
		datura.WithPayload(builder.Segment().Data()),
		datura.WithRole(datura.ArtifactRoleAnswer),
		datura.WithScope(datura.ArtifactScopeContext),
	)
}

func WithArtifact(artifact *datura.Artifact) MessageOption {
	return func(builder *MessageBuilder) {
		payload, err := artifact.DecryptPayload()

		if errnie.Error(err) != nil {
			return
		}

		msgData, err := capnp.Unmarshal(payload)

		if errnie.Error(err) != nil {
			return
		}

		msg, err := ReadRootMessage(msgData)

		if errnie.Error(err) != nil {
			return
		}

		builder.Message = &msg
	}
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

func WithToolCalls(toolCalls ...ToolCall) MessageOption {
	return func(builder *MessageBuilder) {
		toolCallList, err := NewToolCall_List(builder.Segment(), int32(len(toolCalls)))

		if errnie.Error(err) != nil {
			return
		}

		builder.SetToolCalls(toolCallList)
	}
}
