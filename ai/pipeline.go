package ai

import (
	"strings"

	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
)

type Pipeline struct {
	ctx    *Context
	stages []Stage
}

type Stage struct {
	agents     []*Agent
	parallel   bool
	aggregator func([]string) string
}

type StageContext struct {
	originalPrompt string
	currentContext string
}

func NewPipeline(userPrompt string) *Pipeline {
	return &Pipeline{
		ctx: NewContext(userPrompt),
	}
}

func (p *Pipeline) AddSequentialStage(
	agents ...*Agent,
) *Pipeline {
	p.stages = append(p.stages, Stage{
		agents:   agents,
		parallel: false,
	})
	return p
}

func (pipeline *Pipeline) AddParallelStage(
	agents ...*Agent,
) *Pipeline {
	pipeline.stages = append(pipeline.stages, Stage{
		agents:   agents,
		parallel: true,
	})
	return pipeline
}

/*
Execute the pipeline with the given input, orchestrating the flow between stages
*/
func (p *Pipeline) Execute() <-chan provider.Event {
	out := make(chan provider.Event)

	go func() {
		defer close(out)

		for _, stage := range p.stages {
			p.executeSequential(stage, out)
		}
	}()

	return out
}

func (p *Pipeline) executeSequential(stage Stage, out chan<- provider.Event) {
	var outputs []string

	for _, agent := range stage.agents {
		var response strings.Builder

		agent.buffer.Poke(provider.Message{
			Role: "user",
			Content: utils.JoinWith("\n\n",
				p.ctx.Peek(),
				p.ctx.Feedback(agent.name),
				agent.prompt.BuildTask(p.ctx.userPrompt),
			),
		})

		for event := range agent.Generate() {
			out <- event

			if event.Type == provider.EventToken {
				response.WriteString(event.Content)
			}
		}

		outputs = append(outputs, response.String())
		for event := range p.ctx.Poke(response.String()) {
			out <- event
		}
	}
}
