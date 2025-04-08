package context

import (
	"bytes"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

func (ctx Context) Bytes() []byte {
	errnie.Trace("context.Bytes")

	buf := bytes.NewBuffer(nil)

	if _, err := io.Copy(buf, ctx); errnie.Error(err) != nil {
		return nil
	}

	return buf.Bytes()
}
