package toolcall

import (
	"io"

	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ToolCallState uint

const (
	ToolCallStateUninitialized ToolCallState = iota
	ToolCallStateInitialized
	ToolCallStateBuffered
)

/*
Read implements the io.Reader interface for the ToolCall.
It streams the tool call using a Cap'n Proto Encoder.
*/
func (tc *ToolCallBuilder) Read(p []byte) (n int, err error) {
	if tc.State != ToolCallStateBuffered {
		// Buffer is empty, encode current tool call state
		if err = tc.encoder.Encode(tc.ToolCall.Message()); err != nil {
			return 0, errnie.Error(err)
		}

		if err = tc.buffer.Flush(); err != nil {
			return 0, errnie.Error(err)
		}

		tc.State = ToolCallStateBuffered
	}

	return tc.buffer.Read(p)
}

/*
Write implements the io.Writer interface for the ToolCall.
It streams the provided bytes using a Cap'n Proto Decoder.
*/
func (tc *ToolCallBuilder) Write(p []byte) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}

	if n, err = tc.buffer.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	if err = tc.buffer.Flush(); err != nil {
		return n, errnie.Error(err)
	}

	var (
		m   *capnp.Message
		buf ToolCall
	)

	if m, err = tc.decoder.Decode(); err != nil {
		if err == io.EOF {
			// EOF is expected when there's no more data to decode
			return n, nil
		}
		return n, errnie.Error(err)
	}

	if buf, err = ReadRootToolCall(m); err != nil {
		return n, errnie.Error(err)
	}

	tc.ToolCall = &buf
	tc.State = ToolCallStateBuffered
	return n, nil
}

/*
Close implements the io.Closer interface for the ToolCall.
*/
func (tc *ToolCallBuilder) Close() error {
	errnie.Debug("toolcall.Close")

	if err := tc.buffer.Flush(); err != nil {
		return errnie.Error(err)
	}

	tc.buffer = nil
	tc.encoder = nil
	tc.decoder = nil
	tc.ToolCall = nil

	return nil
}
