package core

import (
	"bufio"
	"bytes"
	"encoding/json"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type ConverterData struct {
	Event *Event `json:"event"`
}

type Converter struct {
	*ConverterData
	dec    *json.Decoder
	enc    *json.Encoder
	buffer *bufio.ReadWriter
}

func NewConverter() *Converter {
	buf := bytes.NewBuffer([]byte{})
	buffer := bufio.NewReadWriter(
		bufio.NewReader(buf),
		bufio.NewWriter(buf),
	)

	converter := &Converter{
		ConverterData: &ConverterData{},
		buffer:        buffer,
		dec:           json.NewDecoder(buffer),
		enc:           json.NewEncoder(buffer),
	}

	return converter
}

func (converter *Converter) Read(p []byte) (n int, err error) {
	errnie.Debug("core.Converter.Read")

	if err = converter.buffer.Flush(); err != nil {
		errnie.NewErrIO(err)
		return
	}

	if n, err = converter.buffer.Read(p); err != nil {
		errnie.NewErrIO(err)
	}

	errnie.Debug("core.Converter.Read", "n", n, "err", err)

	return n, err
}

func (converter *Converter) Write(p []byte) (n int, err error) {
	event := &Event{}

	if err = json.Unmarshal(p, event); err != nil {
		errnie.NewErrIO(err)
		return 0, err
	}

	converter.Event = event
	converter.buffer.Discard(converter.buffer.Available())

	if err = converter.enc.Encode(converter.ConverterData); err != nil {
		errnie.NewErrIO(err)
		return 0, err
	}

	errnie.Debug("core.Converter.Write", "n", n, "err", err)

	return len(p), nil
}

func (converter *Converter) Close() error {
	errnie.Debug("core.Converter.Close")

	converter.buffer.Flush()
	converter.ConverterData = nil
	converter.buffer = nil
	converter.dec = nil
	converter.enc = nil

	return nil
}
