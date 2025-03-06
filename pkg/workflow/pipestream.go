package workflow

import (
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type PipeStream struct {
	reader *io.PipeReader
	writer *io.PipeWriter
}

func NewPipeStream() *PipeStream {
	r, w := io.Pipe()
	return &PipeStream{reader: r, writer: w}
}

func (ps *PipeStream) Read(p []byte) (int, error) {
	errnie.Debug("Reading from pipe stream")
	return ps.reader.Read(p)
}

func (ps *PipeStream) Write(p []byte) (int, error) {
	errnie.Debug("Writing to pipe stream", "len", len(p))
	return ps.writer.Write(p)
}

func (ps *PipeStream) Close() error {
	ps.reader.Close()
	ps.writer.Close()
	return nil
}
