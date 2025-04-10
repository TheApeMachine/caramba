package agent

import (
	"bytes"
	"io"
)

func (agent *Agent) Bytes() []byte {
	buf := bytes.NewBuffer(nil)
	io.Copy(buf, agent)
	return buf.Bytes()
}
