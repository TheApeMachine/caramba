package workflow

import (
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type Pipeline struct {
	components []io.ReadWriteCloser
}

/*
NewPipeline creates a pipeline connecting io.ReadWriteCloser components.

It's a convenient wrapper around io.Copy to chain components together.

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
	errnie.Debug("NewPipeline")

	if len(components) < 2 {
		errnie.NewErrValidation("need at least two components")
	}

	return &Pipeline{
		components: components,
	}
}

func (pipeline *Pipeline) Read(p []byte) (n int, err error) {
	for i := range len(pipeline.components) - 1 {
		if _, err = io.Copy(pipeline.components[i+1], pipeline.components[i]); err != nil {
			errnie.NewErrIO(err)
		}
	}

	return pipeline.components[len(pipeline.components)-1].Read(p)
}

func (pipeline *Pipeline) Write(p []byte) (n int, err error) {
	return pipeline.components[0].Write(p)
}

func (pipeline *Pipeline) Close() error {
	var firstErr error

	for _, c := range pipeline.components {
		err := c.Close()
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}
