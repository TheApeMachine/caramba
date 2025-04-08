package message

import (
	"io"

	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Read implements the io.Reader interface for the Message.
It streams the message using a Cap'n Proto Encoder.
*/
func (msg Message) Read(p []byte) (n int, err error) {
	errnie.Trace("message.Read")

	builder := datura.NewRegistry().Get(msg.ID())

	if msg.Is(errnie.StateReady) {
		// Buffer is empty, encode current message state
		if err = builder.Encoder.Encode(msg.Message()); err != nil {
			return 0, errnie.Error(err)
		}

		if err = builder.Buffer.Flush(); err != nil {
			return 0, errnie.Error(err)
		}

		msg.ToState(errnie.StateBusy)
	}

	return builder.Buffer.Read(p)
}

/*
Write implements the io.Writer interface for the Message.
It streams the provided bytes using a Cap'n Proto Decoder.
*/
func (msg Message) Write(p []byte) (n int, err error) {
	errnie.Trace("message.Write")

	builder := datura.NewRegistry().Get(msg.ID())

	if len(p) == 0 {
		return 0, nil
	}

	if n, err = builder.Buffer.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	if err = builder.Buffer.Flush(); err != nil {
		return n, errnie.Error(err)
	}

	var (
		m *capnp.Message
	)

	if m, err = builder.Decoder.Decode(); err != nil {
		if err == io.EOF {
			// EOF is expected when there's no more data to decode
			return n, nil
		}
		return n, errnie.Error(err)
	}

	msg = errnie.Try(ReadRootMessage(m))
	msg.ToState(errnie.StateReady)
	return n, nil
}

/*
Close implements the io.Closer interface for the Message.
*/
func (msg Message) Close() error {
	errnie.Trace("message.Close")

	builder := datura.NewRegistry().Get(msg.ID())

	if err := builder.Buffer.Flush(); err != nil {
		return errnie.Error(err)
	}

	builder.Buffer = nil
	builder.Encoder = nil
	builder.Decoder = nil

	return nil
}
