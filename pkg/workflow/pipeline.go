package workflow

import (
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

// NewPipeline creates a pipeline of ReadWriteCloser components that process data sequentially.
// Data written to the pipeline is processed by each component in order, and the result can
// be read from the returned ReadWriteCloser.
func NewPipeline(components ...io.ReadWriteCloser) io.ReadWriteCloser {
	errnie.Debug("Creating new pipeline", "components", len(components))

	if len(components) < 2 {
		panic("need at least two components")
	}

	// Connect the components in a chain
	for i := 0; i < len(components)-1; i++ {
		source := components[i]
		dest := components[i+1]

		// Copy from each component to the next in sequence
		go func(src, dst io.ReadWriteCloser) {
			io.Copy(dst, src)
		}(source, dest)
	}

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
