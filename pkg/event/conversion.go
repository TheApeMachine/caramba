package event

import (
	"bytes"
	"errors"

	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Encode encodes the artifact to a byte slice.
*/
func (artifact *Artifact) Encode(buf []byte) {
	encoder := capnp.NewEncoder(bytes.NewBuffer(buf))

	if err := encoder.Encode(artifact.Message()); err != nil {
		errnie.Error(err)
	}
}

/*
Decode decodes the artifact from a byte slice.
*/
func (artifact *Artifact) Decode(buf []byte) *Artifact {
	var (
		err     error
		msg     *capnp.Message
		artfct  Artifact
		decoder = capnp.NewDecoder(bytes.NewBuffer(buf))
	)

	if msg, err = decoder.Decode(); err != nil {
		errnie.Error(err)
	}

	if artfct, err = ReadRootArtifact(msg); err != nil {
		errnie.Error(err)
		return nil
	}

	*artifact = artfct
	return artifact
}

/*
Marshal converts the artifact to a byte slice.
*/
func (artifact *Artifact) Marshal(p []byte) {
	var (
		buf []byte
		err error
	)

	if buf, err = artifact.Message().Marshal(); err != nil {
		// Log the error, we don't need to return here, because we
		// rely on the caller handling things if the buffer does not
		// contain the expected data.
		errnie.Error(err)
	}

	// Return the buffer in whatever state it may be.
	copy(p, buf)
}

/*
Unmarshal converts a byte slice to an artifact.
*/
func (artifact *Artifact) Unmarshal(buf []byte) *Artifact {
	var (
		msg    *capnp.Message
		artfct Artifact
		err    error
	)

	if len(buf) == 0 {
		errnie.Error(errors.New("empty buffer"))
		return nil
	}

	if msg, err = capnp.Unmarshal(buf); err != nil {
		errnie.Error(errors.New("capnp unmarshal failed: " + err.Error()))
		return nil
	}

	if msg == nil {
		errnie.Error(errors.New("nil message after unmarshal"))
		return nil
	}

	if artfct, err = ReadRootArtifact(msg); err != nil {
		errnie.Error(errors.New("read root artifact failed: " + err.Error()))
		return nil
	}

	*artifact = artfct
	return artifact
}

func (artifact *Artifact) Pack() []byte {
	var (
		buf []byte
		err error
	)

	if buf, err = artifact.Message().MarshalPacked(); err != nil {
		errnie.Error(err)
	}

	return buf
}

/*
Unpack converts a byte slice to an artifact.
*/
func (artifact *Artifact) Unpack(buf []byte) *Artifact {
	var (
		msg    *capnp.Message
		artfct Artifact
		err    error
	)

	if len(buf) == 0 {
		errnie.Error(errors.New("empty buffer"))
		return nil
	}

	if msg, err = capnp.UnmarshalPacked(buf); err != nil {
		errnie.Error(err)
		return nil
	}

	if msg == nil {
		errnie.Error(errors.New("nil message after unmarshal"))
		return nil
	}

	if artfct, err = ReadRootArtifact(msg); err != nil {
		errnie.Error(err)
		return nil
	}

	*artifact = artfct
	return artifact
}
