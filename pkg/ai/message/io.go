package message

import (
	"io"

	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type MessageState uint

const (
	MessageStateUninitialized MessageState = iota
	MessageStateInitialized
	MessageStateBuffered
)

/*
Read implements the io.Reader interface for the Message.
It streams the message using a Cap'n Proto Encoder.
*/
func (msg *MessageBuilder) Read(p []byte) (n int, err error) {
	errnie.Trace("message.Read")

	if msg.State != MessageStateBuffered {
		// Buffer is empty, encode current message state
		if err = msg.encoder.Encode(msg.Message.Message()); err != nil {
			return 0, errnie.Error(err)
		}

		msg.State = MessageStateBuffered
	}

	if err = msg.buffer.Flush(); err != nil {
		return 0, errnie.Error(err)
	}

	return msg.buffer.Read(p)
}

/*
Write implements the io.Writer interface for the Message.
It streams the provided bytes using a Cap'n Proto Decoder.
*/
func (msg *MessageBuilder) Write(p []byte) (n int, err error) {
	errnie.Trace("message.Write")

	if len(p) == 0 {
		return 0, nil
	}

	if n, err = msg.buffer.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	if err = msg.buffer.Flush(); err != nil {
		return n, errnie.Error(err)
	}

	var (
		m   *capnp.Message
		buf Message
	)

	if m, err = msg.decoder.Decode(); err != nil {
		if err == io.EOF {
			// EOF is expected when there's no more data to decode
			return n, nil
		}
		return n, errnie.Error(err)
	}

	if buf, err = ReadRootMessage(m); err != nil {
		return n, errnie.Error(err)
	}

	msg.Message = &buf
	msg.State = MessageStateBuffered
	return n, nil
}

/*
Close implements the io.Closer interface for the Message.
*/
func (msg *MessageBuilder) Close() error {
	errnie.Trace("message.Close")

	if err := msg.buffer.Flush(); err != nil {
		return errnie.Error(err)
	}

	msg.buffer = nil
	msg.encoder = nil
	msg.decoder = nil
	msg.Message = nil

	return nil
}
