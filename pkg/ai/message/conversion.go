package message

import (
	"bytes"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

func (msg *MessageBuilder) Bytes() []byte {
	buf := bytes.NewBuffer(nil)

	if _, err := io.Copy(buf, msg); errnie.Error(err) != nil {
		return nil
	}

	return buf.Bytes()
}
