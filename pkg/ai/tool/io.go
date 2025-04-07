package tool

import (
	"io"

	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ToolState uint

const (
	ToolStateUninitialized ToolState = iota
	ToolStateInitialized
	ToolStateBuffered
)

/*
Read implements the io.Reader interface for the Message.
It streams the message using a Cap'n Proto Encoder.
*/
func (tb *ToolBuilder) Read(p []byte) (n int, err error) {
	errnie.Trace("tool.Read")

	if tb.State != ToolStateBuffered {
		// Buffer is empty, encode current message state
		if err = tb.encoder.Encode(tb.Message()); err != nil {
			return 0, errnie.Error(err)
		}

		tb.State = ToolStateBuffered
	}

	if err = tb.buffer.Flush(); err != nil {
		return 0, errnie.Error(err)
	}

	return tb.buffer.Read(p)
}

/*
Write implements the io.Writer interface for the Message.
It streams the provided bytes using a Cap'n Proto Decoder.
*/
func (tb *ToolBuilder) Write(p []byte) (n int, err error) {
	errnie.Trace("tool.Write")

	if len(p) == 0 {
		return 0, nil
	}

	if n, err = tb.buffer.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	if err = tb.buffer.Flush(); err != nil {
		return n, errnie.Error(err)
	}

	var (
		m   *capnp.Message
		buf Tool
	)

	if m, err = tb.decoder.Decode(); err != nil {
		if err == io.EOF {
			// EOF is expected when there's no more data to decode
			return n, nil
		}
		return n, errnie.Error(err)
	}

	if buf, err = ReadRootTool(m); err != nil {
		return n, errnie.Error(err)
	}

	tb.Tool = &buf
	tb.State = ToolStateBuffered
	return n, nil
}

/*
Close implements the io.Closer interface for the Message.
*/
func (tb *ToolBuilder) Close() error {
	errnie.Trace("tool.Close")

	if err := tb.buffer.Flush(); err != nil {
		return errnie.Error(err)
	}

	tb.buffer = nil
	tb.encoder = nil
	tb.decoder = nil
	tb.Tool = nil

	return nil
}
