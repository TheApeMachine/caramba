package message

import (
	"errors"
	"fmt"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Read implements the io.Reader interface for the Artifact.
It marshals the entire artifact into the provided byte slice.
*/
func (artifact *Artifact) Read(p []byte) (n int, err error) {
	errnie.Debug("message.Read: Starting read operation")

	buf, err := artifact.Message().Marshal()
	if err != nil {
		errnie.Error(err)
		return 0, err
	}
	errnie.Debug(fmt.Sprintf("message.Read: Marshaled data length: %d", len(buf)))

	// Copy as much as we can into the provided buffer
	n = copy(p, buf)
	errnie.Debug(fmt.Sprintf("message.Read: Copied %d bytes to buffer", n))

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
	if len(p) == 0 {
		errnie.Error(errors.New("empty input"))
		return 0, errnie.Error(errors.New("empty input"))
	}
	errnie.Debug(fmt.Sprintf("message.Write: Starting write operation with %d bytes", len(p)))

	// Use the existing Unmarshal method and check if it succeeded
	if result := artifact.Unmarshal(p); result == nil {
		errnie.Error("failed to unmarshal event")
		return 0, errnie.Error("failed to unmarshal event")
	}
	errnie.Debug("message.Write: Successfully unmarshaled message")

	// Log the state of the artifact after unmarshaling
	id, _ := artifact.Id()
	role, _ := artifact.Role()
	content, _ := artifact.Content()
	errnie.Debug(fmt.Sprintf("message.Write: Artifact after unmarshal - ID: %s, Role: %s, Content length: %d", id, role, len(content)))

	return len(p), nil
}

/*
Close implements the io.Closer interface for the Artifact.
*/
func (artifact *Artifact) Close() error {
	artifact = nil
	return nil
}
