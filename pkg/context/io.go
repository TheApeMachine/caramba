package context

import (
	"errors"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Read implements the io.Reader interface for the Artifact.
It marshals the entire artifact into the provided byte slice.
*/
func (artifact *Artifact) Read(p []byte) (n int, err error) {
	buf := artifact.Marshal()
	p = p[:len(buf)]
	n = copy(p, buf)

	return n, io.EOF
}

/*
Write implements the io.Writer interface for the Artifact.
It unmarshals the provided bytes into the current artifact.
*/
func (artifact *Artifact) Write(p []byte) (n int, err error) {
	if len(p) == 0 {
		return 0, errnie.Error(errors.New("empty input"))
	}

	// Use the existing Unmarshal method and check if it succeeded
	if result := artifact.Unmarshal(p); result == nil {
		return 0, errnie.Error("failed to unmarshal event")
	}

	return len(p), nil
}

/*
Close implements the io.Closer interface for the Artifact.
*/
func (artifact *Artifact) Close() error {
	artifact = nil
	return nil
}
