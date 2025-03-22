package workflow

import "io"

/*
Sink consumes messages without passing through any data.
*/
type Sink struct {
}

func NewSink() *Sink {
	return &Sink{}
}

func (sink *Sink) Read(p []byte) (n int, err error) {
	return 0, io.EOF
}

func (sink *Sink) Write(p []byte) (n int, err error) {
	return len(p), nil
}

func (sink *Sink) Close() error {
	return nil
}
