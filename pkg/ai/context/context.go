package context

import (
	"capnproto.org/go/capnp/v3"
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
