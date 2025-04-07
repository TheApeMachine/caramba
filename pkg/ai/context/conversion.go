package context

import (
	"bytes"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

func (ctx *ContextBuilder) Bytes() []byte {
	buf := bytes.NewBuffer(nil)

	if _, err := io.Copy(buf, ctx); errnie.Error(err) != nil {
		return nil
	}

	return buf.Bytes()
}
