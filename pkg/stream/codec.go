package stream

import (
	"bytes"
	"io"
)

/*
Encoder defines an interface for encoding data into a serialized format.
Implementations of this interface provide the ability to convert Go data structures
into their serialized representations for transmission or storage.
*/
type Encoder interface {
	// Encode serializes the provided value into a format determined by the implementation.
	// Returns an error if the encoding process fails.
	Encode(v any) error
}

/*
Decoder defines an interface for decoding serialized data back into Go data structures.
Implementations of this interface provide the ability to deserialize data from
transmission channels or storage back into usable Go values.
*/
type Decoder interface {
	// Decode deserializes data into the provided value according to the implementation format.
	// Returns an error if the decoding process fails.
	Decode(v any) error
}

/*
Codec combines the Encoder and Decoder interfaces to provide bidirectional data transformation.
It also requires the ability to configure itself with pipe readers and writers for streaming operations.
This interface serves as the foundation for different serialization formats in the streaming system.
*/
type Codec interface {
	Encoder
	Decoder
	// WithPipes configures the codec with input and output pipes for streaming operations.
	// Returns the configured Codec for method chaining.
	WithPipes(*io.PipeReader, *io.PipeWriter) Codec
	WithBuffer(*bytes.Buffer) Codec
}

/*
NewCodec creates and returns a Codec instance of the specified concrete type.
This factory function provides a consistent way to initialize different codec implementations.

Parameters:
  - codecType: A concrete implementation of the Codec interface

Returns the provided codec instance, allowing for method chaining.
*/
func NewCodec(codecType Codec) Codec {
	return codecType
}
