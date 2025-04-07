package toolcall

import (
	"bytes"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

func (tc *ToolCallBuilder) Bytes() []byte {
	buf := bytes.NewBuffer(nil)

	if _, err := io.Copy(buf, tc); errnie.Error(err) != nil {
		return nil
	}

	return buf.Bytes()
}
