package examples

import (
	"io"
	"os"

	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type Pipeline struct {
	agent    *ai.Agent
	provider *provider.OpenAIProvider
	workflow io.ReadWriter
}

func NewPipeline() *Pipeline {
	errnie.Debug("examples.NewPipeline")

	agent := ai.NewAgent()
	provider := provider.NewOpenAIProvider(
		os.Getenv("OPENAI_API_KEY"),
		tweaker.GetEndpoint("openai"),
	)

	pipeline := &Pipeline{
		agent:    agent,
		provider: provider,
		workflow: workflow.NewPipeline(
			agent,
			workflow.NewFeedback(
				provider,
				agent,
			),
		),
	}

	return pipeline
}

func (pipeline *Pipeline) Run() (err error) {
	errnie.Info("Starting pipeline example")

	msg := datura.New(
		datura.WithPayload(provider.NewParams(
			provider.WithMessages(
				provider.NewMessage(
					provider.WithUserRole("Danny", "Write a good programmer joke, but nothing cliche, or knock-knock/chicken-crossed-the-road/etc."),
				),
			),
		).Marshal()),
	)

	if _, err = io.Copy(pipeline, msg); err != nil && err != io.EOF {
		return err
	}

	if _, err = io.Copy(os.Stdout, pipeline); err != nil && err != io.EOF {
		return err
	}

	return nil
}

func (pipeline *Pipeline) Read(p []byte) (n int, err error) {
	errnie.Debug("examples.Pipeline.Read")
	return pipeline.workflow.Read(p)
}

func (pipeline *Pipeline) Write(p []byte) (n int, err error) {
	errnie.Debug("examples.Pipeline.Write")
	return pipeline.workflow.Write(p)
}
