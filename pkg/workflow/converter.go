package workflow

import (
	"bytes"
	"encoding/json"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
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
			var payload []byte

			if payload, err = artifact.DecryptPayload(); err != nil {
				return errnie.Error(err)
			}

			obj := provider.Params{}

			if err = json.Unmarshal(payload, &obj); err != nil {
				return errnie.Error(err)
			}

			out.WriteString(obj.Messages[len(obj.Messages)-1].Content)

			return nil
		}),
		out: out,
	}

	return conv
}

func (c *Converter) Read(p []byte) (n int, err error) {
	return c.out.Read(p)
}

func (c *Converter) Write(p []byte) (n int, err error) {
	return c.buffer.Write(p)
}

func (c *Converter) Close() error {
	return nil
}
