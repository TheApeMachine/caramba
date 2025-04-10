package datura

import (
	"bytes"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

func (artifact *Artifact) Bytes() []byte {
	errnie.Trace("datura.Bytes")

	buf := bytes.NewBuffer([]byte{})

	artifact.ToState(errnie.StateReady)

	if _, err := io.Copy(buf, artifact); errnie.Error(err) != nil {
		return nil
	}

	artifact.ToState(errnie.StateReady)
	return buf.Bytes()
}
