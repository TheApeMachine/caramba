package datura

import (
	"bytes"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

func (artifact *ArtifactBuilder) Bytes() []byte {
	errnie.Trace("datura.Bytes")

	buf := bytes.NewBuffer(nil)

	if _, err := io.Copy(buf, artifact); errnie.Error(err) != nil {
		return nil
	}

	return buf.Bytes()
}
