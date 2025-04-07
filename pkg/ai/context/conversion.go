package context

import (
	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func (ctx *Context) Marshal() ([]byte, error) {
	return ctx.Message().Marshal()
}

/*
Unmarshal a byte slice buffer into a Context type, so we can access its
fields and values.

This is a very cheap operation, because of how Cap 'n Proto works.
*/
func Unmarshal(p []byte) *Context {
	var (
		msg *capnp.Message
		ctx Context
		err error
	)

	// Unmarshal is a bit of a misnomer in the world of Cap 'n Proto,
	// but they went with it anyway.
	if msg, err = capnp.Unmarshal(p); errnie.Error(err) != nil {
		return nil
	}

	// Read a Datagram instance from the message.
	if ctx, err = ReadRootContext(msg); errnie.Error(err) != nil {
		return nil
	}

	return &ctx
}
