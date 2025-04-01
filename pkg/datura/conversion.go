package datura

import (
	"encoding/json"

	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
To is a convenience function to convert the artifact's payload into some
other type by unmarshalling it into the provided type.
*/
func (artifact *Artifact) To(v any) (err error) {
	errnie.Debug("datura.To")

	var payload []byte

	if payload, err = artifact.DecryptPayload(); err != nil {
		return errnie.Error(err, "payload", payload)
	}

	if err = json.Unmarshal(payload, v); err != nil {
		return errnie.Error(err, "payload", payload)
	}

	return nil
}

/*
From is a convenience function to set the artifact's payload from some
other type by marshalling it into the artifact's payload.
*/
func (artifact *Artifact) From(v any) (err error) {
	errnie.Debug("datura.From")

	var payload []byte

	if payload, err = json.Marshal(v); err != nil {
		return errnie.Error(err, "payload", string(payload))
	}

	WithPayload(payload)(artifact)
	return nil
}

/*
Unmarshal a byte slice buffer into a Artifact type, so we can access its
fields and values.

This is a very cheap operation, because of how Cap 'n Proto works.
*/
func Unmarshal(p []byte) *Artifact {
	var (
		msg *capnp.Message
		dg  Artifact
		err error
	)

	// Unmarshal is a bit of a misnomer in the world of Cap 'n Proto,
	// but they went with it anyway.
	if msg, err = capnp.Unmarshal(p); errnie.Error(err) != nil {
		return nil
	}

	// Read a Datagram instance from the message.
	if dg, err = ReadRootArtifact(msg); errnie.Error(err) != nil {
		return nil
	}

	return &dg
}

/*
Error is a convenience function to set an error as the payload of the artifact.
*/
func (artifact *Artifact) Error(e error) (err error) {
	errnie.Debug("datura.Error", "e", e.Error())

	WithError(errnie.New(
		errnie.WithError(e),
	))(artifact)

	return e
}
