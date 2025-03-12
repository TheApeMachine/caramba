package stream

import "io"

type ConversionCodec struct {
	In  Codec
	Out Codec
}

func (codec *ConversionCodec) Encode(v any) error {
	return codec.Out.Encode(v)
}

func (codec *ConversionCodec) Decode(v any) error {
	return codec.In.Decode(v)
}

func (codec *ConversionCodec) WithPipes(pr *io.PipeReader, pw *io.PipeWriter) Codec {
	codec.In.WithPipes(pr, pw)
	codec.Out.WithPipes(pr, pw)
	return codec
}
