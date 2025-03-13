package workflow

import (
	"bytes"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Pipeline manages a chain of io.ReadWriteCloser components.

It connects components together so data written to the pipeline flows through
all components in sequence.
*/
type Pipeline struct {
	components []io.ReadWriteCloser
	buffer     *bytes.Buffer
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

	return &Pipeline{
		components: components,
		buffer:     bytes.NewBuffer([]byte{}),
	}
}

/*
Read implements the io.Reader interface.

It forwards data through the pipeline by copying from each component to the next.
Returns the number of bytes read and any error encountered.
*/
func (pipeline *Pipeline) Read(p []byte) (n int, err error) {
	errnie.Debug("workflow.Pipeline.Read")

	for _, component := range pipeline.components {
		n, err = component.Read(p)

		if err != nil {
			return n, err
		}

		pipeline.buffer.Write(p)
	}

	return n, err
}

/*
Write implements the io.Writer interface.

It writes data to the first component in the pipeline.
Returns the number of bytes written and any error encountered.
*/
func (pipeline *Pipeline) Write(p []byte) (n int, err error) {
	errnie.Debug("workflow.Pipeline.Write", "p", string(p))
	return pipeline.components[0].Write(p)
}

/*
Close implements the io.Closer interface.

It closes all components in the pipeline and collects any errors encountered.
Returns an error if any component failed to close properly.
*/
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
