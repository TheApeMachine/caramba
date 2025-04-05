package context

import (
	"io"

	"capnproto.org/go/capnp/v3"
	"capnproto.org/go/capnp/v3/rpc"
	"github.com/theapemachine/caramba/pkg/ai/message"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ContextBuilder struct {
	Context *Context
}

func New() *ContextBuilder {
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

	ml, err := ctx.NewMessages(int32(1))

	if errnie.Error(err) != nil {
		return nil
	}

	ctx.SetMessages(ml)

	return &ContextBuilder{
		Context: &ctx,
	}
}

func (ctx *ContextBuilder) Add(msg *message.MessageBuilder) *ContextBuilder {
	messages, err := ctx.Context.Messages()
	if errnie.Error(err) != nil {
		return ctx
	}

	ml, err := message.NewMessage_List(ctx.Context.Segment(), int32(messages.Len()+1))
	if errnie.Error(err) != nil {
		return ctx
	}

	for i := range messages.Len() {
		ml.Set(i, messages.At(i))
	}

	ml.Set(messages.Len(), *msg.Message)
	ctx.Context.SetMessages(ml)

	return ctx
}

func (ctx *ContextBuilder) Client() RPC {
	return ContextToClient(ctx.Context)
}

func (ctx *ContextBuilder) Conn(transport io.ReadWriteCloser) *rpc.Conn {
	return rpc.NewConn(rpc.NewStreamTransport(transport), &rpc.Options{
		BootstrapClient: capnp.Client(ctx.Client()),
	})
}
