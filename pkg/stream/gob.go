package stream

import (
	"encoding/gob"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
GobCodec implements the Codec interface using Go's gob encoding.
It provides efficient binary serialization and deserialization of Go values
using Go's native gob format, which is optimized for communication between Go programs.
*/
type GobCodec struct {
	enc *gob.Encoder
	dec *gob.Decoder
}

/*
WithPipes initializes the GobCodec with pipe reader and writer.
It sets up the gob encoder and decoder using the provided pipes for streaming data.

Parameters:
  - pr: Pipe reader used for decoding incoming data
  - pw: Pipe writer used for encoding outgoing data

Returns the configured codec instance for method chaining.
*/
func (codec *GobCodec) WithPipes(pr *io.PipeReader, pw *io.PipeWriter) Codec {
	errnie.Debug("stream.GobCodec.WithPipes")

	codec.enc = gob.NewEncoder(pw)
	codec.dec = gob.NewDecoder(pr)

	return codec
}

/*
Encode serializes the provided value using gob encoding.
It converts the Go data structure into a binary gob representation suitable for transmission.

Parameters:
  - v: The value to be encoded

Returns an error if the encoding process fails.
*/
func (c *GobCodec) Encode(v any) error {
	errnie.Debug("stream.GobCodec.Encode")
	return c.enc.Encode(v)
}

/*
Decode deserializes data into the provided value using gob decoding.
It converts binary gob data back into the original Go data structure.

Parameters:
  - v: Pointer to the value where decoded data will be stored

Returns an error if the decoding process fails, such as type mismatches or data corruption.
*/
func (c *GobCodec) Decode(v any) error {
	errnie.Debug("stream.GobCodec.Decode")
	return c.dec.Decode(v)
}
