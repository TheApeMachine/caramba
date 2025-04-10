package tool

import (
	"bytes"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

func (tool *Tool) Bytes() []byte {
	errnie.Trace("tool.Bytes")

	buf := bytes.NewBuffer(nil)

	if _, err := io.Copy(buf, tool); err != nil {
		return nil
	}

	return buf.Bytes()
}
