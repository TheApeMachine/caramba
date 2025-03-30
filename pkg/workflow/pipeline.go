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
	processed  bool
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

	var nn int64

	if !pipeline.processed {
		for i := range len(pipeline.components) - 1 {
			nn, err = io.Copy(pipeline.components[i+1], pipeline.components[i])
			n += int(nn)

			if err != nil {
				if err == io.EOF {
					continue
				}

				return n, errnie.Error(err)
			}
		}

		pipeline.processed = true
	}

	n, err = pipeline.components[len(pipeline.components)-1].Read(p)

	if err != nil {
		if err == io.EOF {
			pipeline.processed = false
			return n, err
		}

		return n, errnie.Error(err)
	}

	if n == 0 {
		pipeline.processed = false
		return n, io.EOF
	}

	return n, nil
}

/*
Write implements the io.Writer interface.

It writes data to the first component in the pipeline.
Note that writing is optional - components can produce data independently.
*/
func (pipeline *Pipeline) Write(p []byte) (n int, err error) {
	errnie.Debug("workflow.Pipeline.Write")

	if len(pipeline.components) == 0 {
		return len(p), nil
	}

	pipeline.processed = false
	return pipeline.components[0].Write(p)
}

/*
Close implements the io.Closer interface.

It closes all components in the pipeline that implement io.Closer.
*/
func (pipeline *Pipeline) Close() error {
	errnie.Debug("workflow.Pipeline.Close")

	for _, component := range pipeline.components {
		if closer, ok := component.(io.Closer); ok {
			if err := closer.Close(); err != nil {
				return errnie.Error(err)
			}
		}
	}

	return nil
}
