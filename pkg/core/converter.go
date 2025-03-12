package core

import (
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type ConverterData struct {
	Event *Event `json:"event"`
}

type Converter struct {
	*ConverterData
	*stream.Buffer
	pr *io.PipeReader
	pw *io.PipeWriter
}

func NewConverter() *Converter {
	converter := &Converter{
		ConverterData: &ConverterData{},
	}

	converter.pr, converter.pw = io.Pipe()

	converter.Buffer = stream.NewBuffer(
		converter.ConverterData.Event,
		converter.pr,
		func(event any) (err error) {
			converter.Event = event.(*Event)

			go func() {
				defer converter.pw.Close()

				if _, err = converter.pw.Write([]byte(converter.Event.Message.Content)); err != nil {
					err = converter.pw.CloseWithError(err)
				}
			}()

			return err
		},
	).WithCodec(
		stream.NewCodec(&stream.ConversionCodec{
			In:  stream.NewCodec(&stream.GobCodec{}),
			Out: stream.NewCodec(&stream.StringCodec{}),
		}),
	)

	return converter
}

func (converter *Converter) Read(p []byte) (n int, err error) {
	errnie.Debug("core.Converter.Read")
	return converter.Buffer.Read(p)
}

func (converter *Converter) Write(p []byte) (n int, err error) {
	errnie.Debug("core.Converter.Write")
	return converter.Buffer.Write(p)
}

func (converter *Converter) Close() error {
	errnie.Debug("core.Converter.Close")
	return converter.Buffer.Close()
}
