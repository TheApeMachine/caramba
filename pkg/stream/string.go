package stream

import "io"

type StringCodec struct {
	pr *io.PipeReader
	pw *io.PipeWriter
}

func (sc *StringCodec) Encode(v any) error {
	sc.pw.Write(v.([]byte))
	return nil
}

func (sc *StringCodec) Decode(v any) error {
	sc.pr.Read(v.([]byte))
	return nil
}

func (sc *StringCodec) WithPipes(pr *io.PipeReader, pw *io.PipeWriter) Codec {
	sc.pr = pr
	sc.pw = pw
	return sc
}
