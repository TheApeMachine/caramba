package examples

import (
	"io"

	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type Pipeline struct {
	workflow io.ReadWriteCloser
}

func NewPipeline() *Pipeline {
	errnie.Debug("examples.NewPipeline")

	// Create a pipeline with four components
	pipeline := &Pipeline{
		workflow: workflow.NewPipeline(
			ai.NewAgent(),
			provider.NewOpenAIProvider("", ""),
			core.NewConverter(),
		),
	}

	return pipeline
}

func (pipeline *Pipeline) Read(p []byte) (n int, err error) {
	errnie.Debug("examples.Pipeline.Read")

	if n, err = pipeline.workflow.Read(p); err != nil {
		errnie.NewErrIO(err)
	}

	errnie.Debug("examples.Pipeline.Read", "n", n, "err", err)

	return n, err
}

func (pipeline *Pipeline) Write(p []byte) (n int, err error) {
	errnie.Debug("examples.Pipeline.Write", "p", string(p))

	if n, err = pipeline.workflow.Write(p); err != nil {
		errnie.NewErrIO(err)
		return
	}

	errnie.Debug("examples.Pipeline.Write", "n", n, "err", err)

	return n, err
}

func (pipeline *Pipeline) Close() error {
	errnie.Debug("examples.Pipeline.Close")
	return pipeline.workflow.Close()
}
