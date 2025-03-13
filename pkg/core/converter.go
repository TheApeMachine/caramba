package core

import (
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type ConverterData struct {
	Event *Event `json:"event"`
}

type Converter struct {
	*ConverterData
	*stream.Buffer
}

func NewConverter() *Converter {
	converter := &Converter{
		ConverterData: &ConverterData{},
	}

	converter.Buffer = stream.NewBuffer(
		&EventData{},
		converter.ConverterData,
		func(event any) (err error) {
			converter.Event.EventData = event.(*EventData)
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
