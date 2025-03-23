package workflow

import (
	"bytes"

	"github.com/davecgh/go-spew/spew"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Converter struct {
	buffer *stream.Buffer
	out    *bytes.Buffer
}

func NewConverter() *Converter {
	out := bytes.NewBuffer([]byte{})

	conv := &Converter{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("workflow.Converter.buffer.fn")

			spew.Dump(artifact)

			// errnie.Debug("workflow.Converter.buffer.fn", "payload", string(payload))

			// Try to convert to Params if possible
			// buf := &provider.Params{}
			// if err = json.Unmarshal(payload, buf); err == nil && len(buf.Messages) > 0 {
			// 	out.WriteString(buf.Messages[len(buf.Messages)-1].Content)
			// 	return nil
			// }

			// // If not Params, write the raw payload
			// out.Write(payload)
			return nil
		}),
		out: out,
	}

	return conv
}

func (c *Converter) Read(p []byte) (n int, err error) {
	errnie.Debug("workflow.Converter.Read")
	return c.out.Read(p)
}

func (c *Converter) Write(p []byte) (n int, err error) {
	errnie.Debug("workflow.Converter.Write")
	return c.buffer.Write(p)
}

func (c *Converter) Close() error {
	errnie.Debug("workflow.Converter.Close")
	return nil
}
