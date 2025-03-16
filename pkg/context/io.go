package context

import (
	"io"

	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Read implements the io.Reader interface for the Artifact.
It marshals the entire artifact into the provided byte slice.
*/
func (artifact *Artifact) Read(p []byte) (n int, err error) {
	errnie.Debug("event.Read")

	buf, err := artifact.Message().Marshal()

	if err != nil {
		errnie.Error(err)
		return 0, err
	}

	n = copy(p, buf)

	if n < len(buf) {
		errnie.Error(io.ErrShortBuffer)
		return n, io.ErrShortBuffer
	}

	return n, io.EOF
}

/*
Write implements the io.Writer interface for the Artifact.
It unmarshals the provided bytes into the current artifact.
*/
func (artifact *Artifact) Write(p []byte) (n int, err error) {
	errnie.Debug("event.Write")

	var (
		msg *capnp.Message
		buf Artifact
	)

	if msg, err = capnp.Unmarshal(p); err != nil {
		return 0, errnie.Error(err)
	}

	if buf, err = ReadRootArtifact(msg); err != nil {
		return 0, errnie.Error(err)
	}

	*artifact = buf
	return len(p), nil
}

/*
Close implements the io.Closer interface for the Artifact.
*/
func (artifact *Artifact) Close() error {
	artifact = nil
	return nil
}
