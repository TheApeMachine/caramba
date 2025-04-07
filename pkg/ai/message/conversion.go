package message

import (
	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Unmarshal a byte slice buffer into a Artifact type, so we can access its
fields and values.

This is a very cheap operation, because of how Cap 'n Proto works.
*/
func Unmarshal(p []byte) *Message {
	var (
		msg *capnp.Message
		dg  Message
		err error
	)

	// Unmarshal is a bit of a misnomer in the world of Cap 'n Proto,
	// but they went with it anyway.
	if msg, err = capnp.Unmarshal(p); errnie.Error(err) != nil {
		return nil
	}

	// Read a Datagram instance from the message.
	if dg, err = ReadRootMessage(msg); errnie.Error(err) != nil {
		return nil
	}

	return &dg
}
