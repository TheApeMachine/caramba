package workflow

import (
	"encoding/json"
	"io"
	"reflect"

	"github.com/davecgh/go-spew/spew"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type JSONCodec struct {
	stream io.ReadWriteCloser
	dec    *json.Decoder
	enc    *json.Encoder
}

func NewJSONCodec(stream io.ReadWriteCloser) *JSONCodec {
	return &JSONCodec{
		stream: stream,
		dec:    json.NewDecoder(stream),
		enc:    json.NewEncoder(stream),
	}
}

// ReadInto reads from the codec and unmarshals into the provided value
func (jc *JSONCodec) ReadInto(v interface{}) error {
	errnie.Debug("Reading from JSON codec")
	return errnie.Error("JSONCodec read error", "error", jc.dec.Decode(v))
}

// WriteFrom serializes and writes the provided value
func (jc *JSONCodec) WriteFrom(v interface{}) error {
	errnie.Debug("Writing to JSON codec")
	spew.Dump(v)

	return errnie.Error("JSONCodec write error", "error", jc.enc.Encode(v))
}

// Transform reads a value of type In, transforms it, and writes as type Out
func (jc *JSONCodec) Transform(transformFn interface{}) error {
	// Get the function type
	fnType := reflect.TypeOf(transformFn)
	if fnType.Kind() != reflect.Func {
		return errnie.Error("transform must be a function")
	}

	// Create a value of the input type
	inType := fnType.In(0)
	inValue := reflect.New(inType).Interface()

	// Read and unmarshal
	if err := jc.ReadInto(inValue); err != nil {
		return err
	}

	// Call the transform function
	fnValue := reflect.ValueOf(transformFn)
	result := fnValue.Call([]reflect.Value{reflect.ValueOf(inValue).Elem()})

	// Check for error
	if !result[1].IsNil() {
		return result[1].Interface().(error)
	}

	// Write the result
	return jc.WriteFrom(result[0].Interface())
}

func (jc *JSONCodec) Read(v any) error {
	return jc.ReadInto(v)
}

func (jc *JSONCodec) Write(v any) error {
	return jc.WriteFrom(v)
}

func (jc *JSONCodec) Close() error {
	return jc.stream.Close()
}

type CodecStream struct {
	codec *JSONCodec
}
