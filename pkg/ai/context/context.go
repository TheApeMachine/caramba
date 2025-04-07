package context

import (
	"bufio"
	"bytes"
	"io"

	"capnproto.org/go/capnp/v3"
	"capnproto.org/go/capnp/v3/rpc"
	"github.com/theapemachine/caramba/pkg/ai/message"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ContextBuilder struct {
	Context *Context `json:"context"`
	encoder *capnp.Encoder
	decoder *capnp.Decoder
	buffer  *bufio.ReadWriter
	State   ContextState
}

func New() *ContextBuilder {
	errnie.Trace("context.New")

	var (
		arena = capnp.SingleSegment(nil)
		seg   *capnp.Segment
		ctx   Context
		err   error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil
	}

	if ctx, err = NewRootContext(seg); errnie.Error(err) != nil {
		return nil
	}

	// Initialize an empty message list
	ml, err := message.NewMessage_List(seg, 0)
	if errnie.Error(err) != nil {
		return nil
	}

	if errnie.Error(ctx.SetMessages(ml)) != nil {
		return nil
	}

	shared := bytes.NewBuffer(nil)
	buffer := bufio.NewReadWriter(
		bufio.NewReader(shared),
		bufio.NewWriter(shared),
	)

	return &ContextBuilder{
		Context: &ctx,
		encoder: capnp.NewEncoder(buffer),
		decoder: capnp.NewDecoder(buffer),
		buffer:  buffer,
		State:   ContextStateInitialized,
	}
}

func (ctx *ContextBuilder) Add(msg *message.MessageBuilder) *ContextBuilder {
	errnie.Trace("context.Add")

	messages, err := ctx.Context.Messages()
	if errnie.Error(err) != nil {
		return ctx
	}

	// Create a new message list with space for one more message
	ml, err := message.NewMessage_List(ctx.Context.Segment(), int32(messages.Len()+1))
	if errnie.Error(err) != nil {
		return ctx
	}

	// Copy existing messages
	for i := 0; i < messages.Len(); i++ {
		if err := ml.Set(i, messages.At(i)); errnie.Error(err) != nil {
			return ctx
		}
	}

	// Add the new message
	if err := ml.Set(messages.Len(), *msg.Message); errnie.Error(err) != nil {
		return ctx
	}

	// Update the context with the new message list
	if err := ctx.Context.SetMessages(ml); errnie.Error(err) != nil {
		return ctx
	}

	return ctx
}

func (ctx *ContextBuilder) Client() RPC {
	errnie.Trace("context.Client")

	return ContextToClient(ctx.Context)
}

func (ctx *ContextBuilder) Conn(transport io.ReadWriteCloser) *rpc.Conn {
	errnie.Trace("context.Conn")

	return rpc.NewConn(rpc.NewStreamTransport(transport), &rpc.Options{
		BootstrapClient: capnp.Client(ctx.Client()),
	})
}
