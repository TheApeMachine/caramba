package datura

import (
	"io"

	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Read implements the io.Reader interface for the Artifact.
It streams the artifact using a Cap'n Proto Encoder.
*/
func (artifact Artifact) Read(p []byte) (n int, err error) {
	errnie.Trace("artifact.Read")

	builder := NewRegistry().Get(artifact.ID())

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

	return builder.Buffer.Read(p)
}

/*
Write implements the io.Writer interface for the Artifact.
It streams the provided bytes using a Cap'n Proto Decoder.
*/
func (artifact Artifact) Write(p []byte) (n int, err error) {
	errnie.Trace("artifact.Write")

	builder := NewRegistry().Get(artifact.ID())

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
		msg *capnp.Message
	)

	if msg, err = builder.Decoder.Decode(); err != nil {
		if err == io.EOF {
			// EOF is expected when there's no more data to decode
			return n, nil
		}
		return n, errnie.Error(err)
	}

	if artifact, err = ReadRootArtifact(msg); err != nil {
		return n, errnie.Error(err)
	}

	artifact.ToState(errnie.StateReady)
	return n, nil
}

/*
Close implements the io.Closer interface for the Artifact.
*/
func (artifact Artifact) Close() error {
	errnie.Trace("artifact.Close")

	registry := NewRegistry()
	builder := registry.Get(artifact.ID())

	if err := builder.Buffer.Flush(); err != nil {
		return errnie.Error(err)
	}

	builder.Buffer = nil
	builder.Encoder = nil
	builder.Decoder = nil
	registry.Unregister(artifact.ID())

	return nil
}
