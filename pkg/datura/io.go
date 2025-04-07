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
func (artifact *ArtifactBuilder) Read(p []byte) (n int, err error) {
	errnie.Trace("artifact.Read")

	if artifact.state != ArtifactStateBuffered {
		// Buffer is empty, encode current message state
		if err = artifact.encoder.Encode(artifact.Artifact.Message()); err != nil {
			return 0, errnie.Error(err)
		}

		artifact.state = ArtifactStateBuffered
	}

	if err = artifact.buffer.Flush(); err != nil {
		return 0, errnie.Error(err)
	}

	return artifact.buffer.Read(p)
}

/*
Write implements the io.Writer interface for the Artifact.
It streams the provided bytes using a Cap'n Proto Decoder.
*/
func (artifact *ArtifactBuilder) Write(p []byte) (n int, err error) {
	errnie.Trace("artifact.Write")

	if len(p) == 0 {
		return 0, nil
	}

	if n, err = artifact.buffer.Write(p); err != nil {
		return n, errnie.Error(err)
	}

	if err = artifact.buffer.Flush(); err != nil {
		return n, errnie.Error(err)
	}

	var (
		msg *capnp.Message
		buf Artifact
	)

	if msg, err = artifact.decoder.Decode(); err != nil {
		if err == io.EOF {
			// EOF is expected when there's no more data to decode
			return n, nil
		}
		return n, errnie.Error(err)
	}

	if buf, err = ReadRootArtifact(msg); err != nil {
		return n, errnie.Error(err)
	}

	artifact.Artifact = &buf
	artifact.state = ArtifactStateBuffered
	return n, nil
}

/*
Close implements the io.Closer interface for the Artifact.
*/
func (artifact *ArtifactBuilder) Close() error {
	errnie.Trace("artifact.Close")

	if err := artifact.buffer.Flush(); err != nil {
		return errnie.Error(err)
	}

	artifact.buffer = nil
	artifact.encoder = nil
	artifact.decoder = nil
	artifact.Artifact = nil

	return nil
}
