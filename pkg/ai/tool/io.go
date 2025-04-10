package tool

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
func (tool *Tool) Read(p []byte) (n int, err error) {
	errnie.Trace("tool.Read")

	builder := datura.NewRegistry().Get(tool)

	if tool.Is(errnie.StateReady) {
		// Buffer is empty, encode current message state
		if err = builder.Encoder.Encode(tool.Message()); err != nil {
			return 0, errnie.Error(err)
		}

		if err = builder.Buffer.Flush(); err != nil {
			return 0, errnie.Error(err)
		}

		tool.ToState(errnie.StateBusy)
	}

	if !tool.Is(errnie.StateBusy) {
		return 0, errnie.New(
			errnie.WithError(errors.New("bad read state")),
			errnie.WithMessage("tool is not busy"),
		)
	}

	return builder.Buffer.Read(p)
}

/*
Write implements the io.Writer interface for the Context.
It streams the provided bytes using a Cap'n Proto Decoder.
*/
func (tool *Tool) Write(p []byte) (n int, err error) {
	errnie.Trace("tool.Write")

	tool.ToState(errnie.StateBusy)

	builder := datura.NewRegistry().Get(tool)

	if len(p) == 0 {
		return 0, nil
	}

	if n, err = builder.Buffer.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	var (
		msg *capnp.Message
		buf Tool
	)

	if msg, err = builder.Decoder.Decode(); err != nil {
		if err == io.EOF {
			// EOF is expected when there's no more data to decode
			return n, nil
		}
		return n, errnie.Error(err)
	}

	if buf, err = ReadRootTool(msg); err != nil {
		return n, errnie.Error(err)
	}

	*tool = buf
	tool.ToState(errnie.StateReady)
	return n, nil
}

/*
Close implements the io.Closer interface for the Context.
*/
func (tool *Tool) Close() error {
	errnie.Trace("tool.Close")

	builder := datura.NewRegistry().Get(tool)

	if err := builder.Buffer.Flush(); err != nil {
		return errnie.Error(err)
	}

	builder.Buffer = nil
	builder.Encoder = nil
	builder.Decoder = nil
	datura.NewRegistry().Unregister(tool)

	return nil
}
