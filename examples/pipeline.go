package examples

import (
	"io"

	"github.com/spf13/viper"
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
	errnie.Debug("NewPipeline")

	return &Pipeline{
		workflow: workflow.NewPipeline(
			core.NewEvent(
				core.NewMessage("user", "Danny", viper.GetViper().GetString("tests.joke")),
				nil,
			),
			ai.NewAgent(),
			provider.NewOpenAIProvider("", ""),
		),
	}
}

func (pipeline *Pipeline) Read(p []byte) (n int, err error) {
	errnie.Debug("Pipeline.Read")
	return pipeline.workflow.Read(p)
}

func (pipeline *Pipeline) Write(p []byte) (n int, err error) {
	errnie.Debug("Pipeline.Write", "p", string(p))
	return pipeline.workflow.Write(p)
}

func (pipeline *Pipeline) Close() error {
	errnie.Debug("Pipeline.Close")
	return pipeline.workflow.Close()
}
