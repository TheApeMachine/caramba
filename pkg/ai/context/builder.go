package context

import (
	"capnproto.org/go/capnp/v3"
	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/ai/message"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func New() Context {
	errnie.Trace("context.New")

	var (
		arena = capnp.SingleSegment(nil)
		seg   *capnp.Segment
		ctx   Context
		err   error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return errnie.Try(NewContext(seg)).ToState(errnie.StateError)
	}

	if ctx, err = NewRootContext(seg); errnie.Error(err) != nil {
		return errnie.Try(NewContext(seg)).ToState(errnie.StateError)
	}

	ctx.SetUuid(uuid.New().String())
	ctx.SetState(uint64(errnie.StatePending))

	// Initialize an empty message list
	ml, err := message.NewMessage_List(seg, 0)
	if errnie.Error(err) != nil {
		return errnie.Try(NewContext(seg)).ToState(errnie.StateError)
	}

	if errnie.Error(ctx.SetMessages(ml)) != nil {
		return errnie.Try(NewContext(seg)).ToState(errnie.StateError)
	}

	return datura.Register(ctx)
}

func (ctx Context) Add(msg message.Message) Context {
	errnie.Trace("context.Add")

	messages := errnie.Try(ctx.Messages())

	// Create a new message list with space for one more message
	ml, err := message.NewMessage_List(ctx.Segment(), int32(messages.Len()+1))
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
	if err := ml.Set(messages.Len(), msg); errnie.Error(err) != nil {
		return ctx
	}

	// Update the context with the new message list
	if err := ctx.SetMessages(ml); errnie.Error(err) != nil {
		return ctx
	}

	return ctx
}
