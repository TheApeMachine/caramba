package stream

import (
	"github.com/theapemachine/caramba/pkg/datura"
)

type Converter struct {
	buffer *Buffer
}

func NewConverter() *Converter {
	conv := &Converter{
		buffer: NewBuffer(func(artifact *datura.Artifact) (err error) {
			return nil
		}),
	}

	return conv
}

func (c *Converter) Read(p []byte) (n int, err error) {
	return c.buffer.Read(p)
}

func (c *Converter) Write(p []byte) (n int, err error) {
	return c.buffer.Write(p)
}

func (c *Converter) Close() error {
	return nil
}
