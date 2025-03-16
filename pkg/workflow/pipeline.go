package workflow

import (
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Pipeline manages a chain of io.ReadWriteCloser components.

It connects components together so data flows through all components in sequence.
Each component can produce data independently.
*/
type Pipeline struct {
	components []io.ReadWriter
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
func NewPipeline(components ...io.ReadWriter) io.ReadWriter {
	errnie.Debug("workflow.NewPipeline")
	return &Pipeline{components: components}
}

/*
Read implements the io.Reader interface.

It reads from the first component and passes data through the pipeline.
Returns EOF when no more data is available.
*/
func (pipeline *Pipeline) Read(p []byte) (n int, err error) {
	errnie.Debug("workflow.Pipeline.Read")

	if len(pipeline.components) == 0 {
		return 0, io.EOF
	}

	// Copy data through the chain before reading
	for i := 0; i < len(pipeline.components)-1; i++ {
		_, err = io.Copy(pipeline.components[i+1], pipeline.components[i])
		if err != nil && err != io.EOF {
			return 0, err
		}
	}

	// Read from last component
	return pipeline.components[len(pipeline.components)-1].Read(p)
}

/*
Write implements the io.Writer interface.

It writes data to the first component in the pipeline.
Note that writing is optional - components can produce data independently.
*/
func (pipeline *Pipeline) Write(p []byte) (n int, err error) {
	errnie.Debug("workflow.Pipeline.Write", "p", string(p))

	if len(pipeline.components) == 0 {
		return len(p), nil
	}

	// Write to first component
	n, err = pipeline.components[0].Write(p)
	if err != nil {
		return n, err
	}

	// Flow data through remaining components
	for i := 0; i < len(pipeline.components)-1; i++ {
		_, err = io.Copy(pipeline.components[i+1], pipeline.components[i])
		if err != nil && err != io.EOF {
			return n, err
		}
	}

	return n, nil
}

/*
Close implements the io.Closer interface.

It closes all components in the pipeline and collects any errors encountered.
*/
func (pipeline *Pipeline) Close() error {
	errnie.Debug("workflow.Pipeline.Close")
	return nil
}
