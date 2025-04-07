package toolcall

import (
	"bufio"
	"bytes"

	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ToolCallBuilder struct {
	ToolCall *ToolCall
	encoder  *capnp.Encoder
	decoder  *capnp.Decoder
	buffer   *bufio.ReadWriter
	State    ToolCallState
}

type ToolCallOption func(*ToolCallBuilder)

func New(opts ...ToolCallOption) *ToolCallBuilder {
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

	shared := bytes.NewBuffer(nil)
	buffer := bufio.NewReadWriter(
		bufio.NewReader(shared),
		bufio.NewWriter(shared),
	)

	toolCallBuilder := &ToolCallBuilder{
		ToolCall: &toolCall,
		encoder:  capnp.NewEncoder(buffer),
		decoder:  capnp.NewDecoder(buffer),
		buffer:   buffer,
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
