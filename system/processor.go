package system

import "io"

type Processor struct {
	io.ReadWriteCloser
}

func NewProcessor() *Processor {
	return &Processor{}
}

func (processor *Processor) Read(p []byte) (n int, err error) {
	return processor.ReadWriteCloser.Read(p)
}

func (processor *Processor) Write(p []byte) (n int, err error) {
	return processor.ReadWriteCloser.Write(p)
}

func (processor *Processor) Close() error {
	return processor.ReadWriteCloser.Close()
}
