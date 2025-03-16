package tool

import (
	"io"
)

/*
Read implements the io.Reader interface for the Artifact.
It marshals the entire artifact into the provided byte slice.
*/
func (artifact *Artifact) Read(p []byte) (n int, err error) {
	buf := artifact.Marshal()

	// Grow p to the size of buf.
	p = p[:len(buf)]

	// Copy buf to p.
	copy(p, buf)

	// Return the number of bytes copied and an EOF error.
	return len(buf), io.EOF
}

/*
Write implements the io.Writer interface for the Artifact.
It unmarshals the provided bytes into the current artifact.
*/
func (artifact *Artifact) Write(p []byte) (n int, err error) {
	artifact.Unmarshal(p)
	return len(p), nil
}

/*
Close implements the io.Closer interface for the Artifact.
*/
func (artifact *Artifact) Close() error {
	artifact = nil
	return nil
}
