package toolcall

import (
	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ToolCallBuilder struct {
	ToolCall *ToolCall
}

type ToolCallOption func(*ToolCallBuilder)

func NewToolCallBuilder(opts ...ToolCallOption) *ToolCallBuilder {
	var (
		arena    = capnp.SingleSegment(nil)
		seg      *capnp.Segment
		toolCall ToolCall
		err      error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil
	}

	if toolCall, err = NewRootToolCall(seg); errnie.Error(err) != nil {
		return nil
	}

	toolCallBuilder := &ToolCallBuilder{
		ToolCall: &toolCall,
	}

	for _, opt := range opts {
		opt(toolCallBuilder)
	}

	return toolCallBuilder
}

func WithID(id string) ToolCallOption {
	return func(toolCallBuilder *ToolCallBuilder) {
		toolCallBuilder.ToolCall.SetId(id)
	}
}

func WithName(name string) ToolCallOption {
	return func(toolCallBuilder *ToolCallBuilder) {
		toolCallBuilder.ToolCall.SetName(name)
	}
}

func WithArgs(args string) ToolCallOption {
	return func(toolCallBuilder *ToolCallBuilder) {
		if err := toolCallBuilder.ToolCall.SetArguments(args); errnie.Error(err) != nil {
			return
		}
	}
}
