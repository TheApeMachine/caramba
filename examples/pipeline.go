package examples

import (
	"io"
	"os"

	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type Pipeline struct {
	workflow io.ReadWriter
}

func NewPipeline() *Pipeline {
	errnie.Debug("examples.NewPipeline")

	pipeline := &Pipeline{
		workflow: workflow.NewPipeline(
			ai.NewAgent(),
			provider.NewOpenAIProvider(
				os.Getenv("OPENAI_API_KEY"),
				tweaker.GetEndpoint("openai"),
			),
		),
	}

	return pipeline
}

func (pipeline *Pipeline) Read(p []byte) (n int, err error) {
	errnie.Debug("examples.Pipeline.Read")
	return pipeline.workflow.Read(p)
}

func (pipeline *Pipeline) Write(p []byte) (n int, err error) {
	errnie.Debug("examples.Pipeline.Write")
	return pipeline.workflow.Write(p)
}
