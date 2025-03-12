package workflow

import (
	"bufio"
	"bytes"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

// Pipeline is a component that chains multiple ReadWriteCloser components together
// into a processing pipeline where data flows from the first component to the last.
type Pipeline struct {
	components []io.ReadWriteCloser
	buffer     *bufio.ReadWriter
}

/*
NewPipeline creates a pipeline connecting io.ReadWriteCloser components.

It connects components together so data written to the pipeline flows through
all components in sequence.

Example:

	// Simple pipeline
	p := workflow.NewPipeline(message, agent, provider)
	io.Copy(os.Stdout, p)

	// Nested pipelines
	p1 := workflow.NewPipeline(message, agent, provider)
	p2 := workflow.NewPipeline(message, agent, provider, p1)
	io.Copy(os.Stdout, p2)
*/
func NewPipeline(components ...io.ReadWriteCloser) io.ReadWriteCloser {
	errnie.Debug("workflow.NewPipeline")

	buf := bytes.NewBuffer([]byte{})

	pipeline := &Pipeline{
		components: components,
		buffer: bufio.NewReadWriter(
			bufio.NewReader(buf),
			bufio.NewWriter(buf),
		),
	}

	return pipeline
}

func (pipeline *Pipeline) Read(p []byte) (n int, err error) {
	errnie.Debug("workflow.Pipeline.Read")
	var nn int64

	for i := range len(pipeline.components) - 1 {
		if err = pipeline.buffer.Flush(); err != nil {
			errnie.NewErrIO(err)
			return
		}

		nn, err = io.Copy(pipeline.components[i+1], pipeline.components[i])

		if err != nil && err != io.EOF {
			errnie.Unwrap(err).Add(err)
			return
		}

		n += int(nn)

		errnie.Debug("workflow.Pipeline.Read", "nn", nn, "err", err)
	}

	return n, err
}

func (pipeline *Pipeline) Write(p []byte) (n int, err error) {
	errnie.Debug("workflow.Pipeline.Write", "p", string(p))

	if n, err = pipeline.buffer.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	errnie.Debug("workflow.Pipeline.Write", "n", n, "err", err)

	return pipeline.components[0].Write(p)
}

func (pipeline *Pipeline) Close() error {
	errnie.Debug("workflow.Pipeline.Close")
	err := errnie.NewError(nil)

	// Close all pipes used for connections
	for _, closer := range pipeline.components {
		if e := closer.Close(); e != nil {
			errnie.Unwrap(err).Add(e)
		}
	}

	return err
}
