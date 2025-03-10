package core

import (
	"bytes"
	"encoding/json"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type ConverterData struct {
	Event *Event `json:"event"`
}

type Converter struct {
	*ConverterData
	dec *json.Decoder
	in  *bytes.Buffer
	out *bytes.Buffer
}

func NewConverter() *Converter {
	in := bytes.NewBuffer([]byte{})
	out := bytes.NewBuffer([]byte{})

	return &Converter{
		ConverterData: &ConverterData{},
		dec:           json.NewDecoder(in),
		in:            in,
		out:           out,
	}
}

func (converter *Converter) Read(p []byte) (n int, err error) {
	return converter.out.Read(p)
}

func (converter *Converter) Write(p []byte) (n int, err error) {
	errnie.Debug("Agent.Write", "p", string(p))

	// Reset the output buffer whenever we write new data
	if converter.out.Len() > 0 {
		converter.out.Reset()
	}

	// Write the incoming bytes to the input buffer
	n, err = converter.in.Write(p)
	if err != nil {
		return n, err
	}

	// Try to decode the data from the input buffer
	// If it fails, we still return the bytes written but keep the error
	event := NewEvent(nil, nil)

	if decErr := converter.dec.Decode(&event); decErr == nil {
		if event.Message == nil {
			return n, errnie.NewErrValidation("message is required")
		}

		// Only update if decoding was successful
		if _, err = converter.out.WriteString(event.Message.Content); err != nil {
			return n, errnie.NewErrIO(err)
		}
	}

	return n, nil
}

func (converter *Converter) Close() error {
	return nil
}
