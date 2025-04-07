package context

import (
	"io"

	capnp "capnproto.org/go/capnp/v3"
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
func (ctx *ContextBuilder) Read(p []byte) (n int, err error) {
	if ctx.State != ContextStateBuffered {
		// Buffer is empty, encode current message state
		if err = ctx.encoder.Encode(ctx.Context.Message()); err != nil {
			return 0, errnie.Error(err)
		}

		if err = ctx.buffer.Flush(); err != nil {
			return 0, errnie.Error(err)
		}

		ctx.State = ContextStateBuffered
	}

	return ctx.buffer.Read(p)
}

/*
Write implements the io.Writer interface for the Context.
It streams the provided bytes using a Cap'n Proto Decoder.
*/
func (ctx *ContextBuilder) Write(p []byte) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}

	if n, err = ctx.buffer.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	if err = ctx.buffer.Flush(); err != nil {
		return n, errnie.Error(err)
	}

	var (
		msg *capnp.Message
		buf Context
	)

	if msg, err = ctx.decoder.Decode(); err != nil {
		if err == io.EOF {
			// EOF is expected when there's no more data to decode
			return n, nil
		}
		return n, errnie.Error(err)
	}

	if buf, err = ReadRootContext(msg); err != nil {
		return n, errnie.Error(err)
	}

	ctx.Context = &buf
	ctx.State = ContextStateBuffered
	return n, nil
}

/*
Close implements the io.Closer interface for the Context.
*/
func (ctx *ContextBuilder) Close() error {
	errnie.Debug("context.Close")

	if err := ctx.buffer.Flush(); err != nil {
		return errnie.Error(err)
	}

	ctx.buffer = nil
	ctx.encoder = nil
	ctx.decoder = nil
	ctx.Context = nil

	return nil
}
