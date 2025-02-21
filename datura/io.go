package datura

import "io"

func (artifact *Artifact) Read(p []byte) (n int, err error) {
	copy(p, artifact.Encode())
	return len(p), io.EOF
}

func (artifact *Artifact) Write(p []byte) (n int, err error) {
	artifact.Decode(p)
	return len(p), nil
}

func (artifact *Artifact) Close() error {
	return nil
}
