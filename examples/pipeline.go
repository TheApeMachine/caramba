package examples

import (
	"io"
	"os"

	"github.com/davecgh/go-spew/spew"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/event"
	"github.com/theapemachine/caramba/pkg/message"
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

	msg := message.New(
		message.UserRole,
		"Danny",
		"Write a good programmer joke, but nothing cliche, or knock-knock/chicken-crossed-the-road/etc.",
	)

	msg2, err := msg.Message().Marshal()
	if errnie.Error(err) != nil {
		return err
	}

	evt := event.New(
		"example.pipeline",
		event.MessageEvent,
		event.UserRole,
		msg2,
	)

	if _, err = io.Copy(pipeline, evt); err != nil && err != io.EOF {
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

func (pipeline *Pipeline) AgentContext() {
	ctx := &context.Artifact{}

	io.Copy(ctx, pipeline.agent)

	messages, err := ctx.Messages()

	if errnie.Error(err) != nil {
		errnie.Error(err)
	}

	for idx := range messages.Len() {
		content, err := messages.At(idx).Content()

		if errnie.Error(err) != nil {
			errnie.Error(err)
		}

		spew.Dump(content)
	}
}
