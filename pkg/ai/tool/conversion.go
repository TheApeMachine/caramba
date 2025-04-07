package tool

import (
	"bytes"
	"io"
)

func (tb *ToolBuilder) Bytes() []byte {
	buf := bytes.NewBuffer(nil)

	if _, err := io.Copy(buf, tb); err != nil {
		return nil
	}

	return buf.Bytes()
}
