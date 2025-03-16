package event

import (
	"fmt"
	"io"

	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Read implements the io.Reader interface for the Artifact.
It marshals the entire artifact into the provided byte slice.
*/
func (artifact *Artifact) Read(p []byte) (n int, err error) {
	errnie.Debug("event.Read: Starting read operation")

	buf, err := artifact.Message().Marshal()
	if err != nil {
		errnie.Error(err)
		return 0, err
	}
	errnie.Debug(fmt.Sprintf("event.Read: Marshaled data length: %d", len(buf)))

	// Copy as much as we can into the provided buffer
	n = copy(p, buf)
	errnie.Debug(fmt.Sprintf("event.Read: Copied %d bytes to buffer", n))

	// If we couldn't copy everything, return ErrShortBuffer
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
	errnie.Debug(fmt.Sprintf("event.Write: Starting write operation with %d bytes", len(p)))

	var (
		msg *capnp.Message
		buf Artifact
	)

	// First unmarshal the incoming data
	if msg, err = capnp.Unmarshal(p); err != nil {
		errnie.Error(err)
		return 0, errnie.Error(err)
	}
	errnie.Debug("event.Write: Successfully unmarshaled message")

	if buf, err = ReadRootArtifact(msg); err != nil {
		errnie.Error(err)
		return 0, errnie.Error(err)
	}
	errnie.Debug("event.Write: Successfully read root artifact")

	// Log the state of the artifact before and after copying
	id, _ := buf.Id()
	typ, _ := buf.Type()
	payload, _ := buf.Payload()
	errnie.Debug(fmt.Sprintf("event.Write: Artifact before copy - ID: %s, Type: %s, Payload length: %d", id, typ, len(payload)))

	// Copy the entire struct from buf to artifact
	*artifact = buf

	// Verify the copy worked
	id, _ = artifact.Id()
	typ, _ = artifact.Type()
	payload, _ = artifact.Payload()
	errnie.Debug(fmt.Sprintf("event.Write: Artifact after copy - ID: %s, Type: %s, Payload length: %d", id, typ, len(payload)))

	return len(p), nil
}

/*
Close implements the io.Closer interface for the Artifact.
*/
func (artifact *Artifact) Close() error {
	artifact = nil
	return nil
}
