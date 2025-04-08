package context

import (
	"errors"
	"io"

	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ContextState uint

const (
	ContextStateUninitialized ContextState = iota
	ContextStateInitialized
	ContextStateBuffered
)

/*
Read implements the io.Reader interface for the Context.
It streams the context using a Cap'n Proto Encoder.
*/
func (ctx *Context) Read(p []byte) (n int, err error) {
	errnie.Trace("context.Read")

	builder := datura.NewRegistry().Get(ctx)

	if ctx.Is(errnie.StateReady) {
		// Buffer is empty, encode current message state
		if err = builder.Encoder.Encode(ctx.Message()); err != nil {
			return 0, errnie.Error(err)
		}

		if err = builder.Buffer.Flush(); err != nil {
			return 0, errnie.Error(err)
		}

		ctx.ToState(errnie.StateBusy)
	}

	if !ctx.Is(errnie.StateBusy) {
		return 0, errnie.New(
			errnie.WithError(errors.New("bad read state")),
			errnie.WithMessage("context is not busy"),
		)
	}

	return builder.Buffer.Read(p)
}

/*
Write implements the io.Writer interface for the Context.
It streams the provided bytes using a Cap'n Proto Decoder.
*/
func (ctx *Context) Write(p []byte) (n int, err error) {
	errnie.Trace("context.Write")

	ctx.ToState(errnie.StateBusy)

	builder := datura.NewRegistry().Get(ctx)

	if len(p) == 0 {
		return 0, nil
	}

	if n, err = builder.Buffer.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	var (
		msg *capnp.Message
		buf Context
	)

	if msg, err = builder.Decoder.Decode(); err != nil {
		if err == io.EOF {
			// EOF is expected when there's no more data to decode
			return n, nil
		}
		return n, errnie.Error(err)
	}

	if buf, err = ReadRootContext(msg); err != nil {
		return n, errnie.Error(err)
	}

	*ctx = buf
	ctx.ToState(errnie.StateReady)
	return n, nil
}

/*
Close implements the io.Closer interface for the Context.
*/
func (ctx *Context) Close() error {
	errnie.Trace("context.Close")

	builder := datura.NewRegistry().Get(ctx)

	if err := builder.Buffer.Flush(); err != nil {
		return errnie.Error(err)
	}

	builder.Buffer = nil
	builder.Encoder = nil
	builder.Decoder = nil
	datura.NewRegistry().Unregister(ctx)

	return nil
}
