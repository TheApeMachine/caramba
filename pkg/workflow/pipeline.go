package workflow

import (
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

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
		errnie.NewError(errnie.NewErrValidation("need at least two components"))
	}

	// Connect the components in a chain
	for i := range len(components)-1 {
		source := components[i]
		dest := components[i+1]

		// Copy from each component to the next in sequence
		go func(src, dst io.ReadWriteCloser) {
			_, err := io.Copy(dst, src)
			if err != nil {
				errnie.NewError(errnie.NewErrIO(err))
			}
		}(source, dest)
	}

	// Return a composite ReadWriteCloser
	return struct {
		io.Reader
		io.Writer
		io.Closer
	}{
		Reader: components[len(components)-1],
		Writer: components[0],
		Closer: closerFunc(func() error {
			for _, c := range components {
				c.Close()
			}
			return nil
		}),
	}
}

type closerFunc func() error

func (fn closerFunc) Close() error { return fn() }
