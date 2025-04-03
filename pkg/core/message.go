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
	errnie.Debug("core.MessageBuilder.Artifact")

	data, err := builder.Message.Message().Marshal()

	if errnie.Error(err) != nil {
		return nil
	}

	return datura.New(
		datura.WithPayload(data),
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
		if errnie.Error(builder.SetRole(role)) != nil {
			return
		}
	}
}

func WithName(name string) MessageOption {
	return func(builder *MessageBuilder) {
		if errnie.Error(builder.SetName(name)) != nil {
			return
		}
	}
}

func WithContent(content string) MessageOption {
	return func(builder *MessageBuilder) {
		if errnie.Error(builder.SetContent(content)) != nil {
			return
		}
	}
}

func WithToolCalls(toolCalls ...ToolCall) MessageOption {
	return func(builder *MessageBuilder) {
		toolCallList, err := NewToolCall_List(builder.Segment(), int32(len(toolCalls)))

		if errnie.Error(err) != nil {
			return
		}

		if errnie.Error(builder.SetToolCalls(toolCallList)) != nil {
			return
		}
	}
}
