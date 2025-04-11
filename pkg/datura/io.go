package datura

import (
	"errors"
	"io"

	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Read implements the io.Reader interface for the Artifact.
It streams the artifact using a Cap'n Proto Encoder.
*/
func (artifact *Artifact) Read(p []byte) (n int, err error) {
	builder := NewRegistry().Get(artifact)

	if artifact.Is(errnie.StateReady) {
		// Buffer is empty, encode current message state
		if err = builder.Encoder.Encode(artifact.Message()); err != nil {
			return 0, errnie.Error(err)
		}

		if err = builder.Buffer.Flush(); err != nil {
			return 0, errnie.Error(err)
		}

		artifact.ToState(errnie.StateBusy)
	}

	if !artifact.Is(errnie.StateBusy) {
		return 0, errnie.New(
			errnie.WithError(errors.New("bad read state")),
			errnie.WithMessage("artifact is not busy"),
		)
	}

	return builder.Buffer.Read(p)
}

/*
Write implements the io.Writer interface for the Artifact.
It streams the provided bytes using a Cap'n Proto Decoder.
*/
func (artifact *Artifact) Write(p []byte) (n int, err error) {
	artifact.ToState(errnie.StateBusy)

	builder := NewRegistry().Get(artifact)

	if n, err = builder.Buffer.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	if err = builder.Buffer.Flush(); err != nil {
		return n, errnie.Error(err)
	}

	var (
		msg *capnp.Message
		art Artifact
	)

	if msg, err = builder.Decoder.Decode(); err != nil {
		if err == io.EOF {
			// EOF is expected when there's no more data to decode
			return n, nil
		}
		return n, errnie.Error(err)
	}

	if art, err = ReadRootArtifact(msg); errnie.Error(err) != nil {
		return n, errnie.Error(err)
	}

	artifact = &art
	artifact.ToState(errnie.StateReady)
	return n, nil
}

/*
Close implements the io.Closer interface for the Artifact.
*/
func (artifact *Artifact) Close() error {
	registry := NewRegistry()
	builder := registry.Get(artifact)

	if err := builder.Buffer.Flush(); err != nil {
		return errnie.Error(err)
	}

	builder.Buffer = nil
	builder.Encoder = nil
	builder.Decoder = nil
	registry.Unregister(artifact)

	return nil
}
