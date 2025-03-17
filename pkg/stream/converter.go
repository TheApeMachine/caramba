package stream

import (
	"bytes"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/event"
	"github.com/theapemachine/caramba/pkg/message"
)

type Converter struct {
	buffer *Buffer
	reader *bytes.Buffer
}

func NewConverter() *Converter {
	conv := &Converter{
		reader: bytes.NewBuffer([]byte{}),
	}

	conv.buffer = NewBuffer(func(evt *event.Artifact) error {
		payload, err := evt.Payload()
		if err != nil {
			return errnie.Error(err)
		}

		msg := &message.Artifact{}
		msg.Write(payload)

		content, err := msg.Content()

		if err != nil {
			return errnie.Error(err)
		}

		if _, err = conv.reader.WriteString(content); err != nil {
			return errnie.Error(err)
		}

		return nil
	})

	return conv
}

func (c *Converter) Read(p []byte) (n int, err error) {
	return c.reader.Read(p)
}

func (c *Converter) Write(p []byte) (n int, err error) {
	return c.buffer.Write(p)
}

func (c *Converter) Close() error {
	return nil
}
