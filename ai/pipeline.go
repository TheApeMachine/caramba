package ai

import (
	"github.com/theapemachine/caramba/provider"
)

/*
Pipeline is a sequence of stages, which are executed sequentially.
*/
type Pipeline struct {
	ctx    *Context
	stages []Stage
}

/*
Stage is a sequence of agents, which are executed sequentially.
*/
type Stage struct {
	agents     []*Agent
	parallel   bool
	aggregator func([]string) string
}

/*
StageContext is the context for a stage, which is used to pass information
between stages.
*/
type StageContext struct {
	originalPrompt string
	currentContext string
}

/*
NewPipeline creates a new pipeline, with the given user prompt.
*/
func NewPipeline(userPrompt string) *Pipeline {
	return &Pipeline{
		ctx: NewContext(userPrompt),
	}
}

/*
AddSequentialStage adds a new sequential stage to the pipeline.
*/
func (p *Pipeline) AddSequentialStage(
	agents ...*Agent,
) *Pipeline {
	p.stages = append(p.stages, Stage{
		agents:   agents,
		parallel: false,
	})
	return p
}

/*
AddParallelStage adds a new parallel stage to the pipeline.
*/
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
func (pipeline *Pipeline) Execute() <-chan provider.Event {
	out := make(chan provider.Event)

	go func() {
		defer close(out)

		for _, stage := range pipeline.stages {
			for _, agent := range stage.agents {
				pipeline.ctx.SetCurrentAgent(agent)

				for event := range pipeline.ctx.Generate() {
					out <- event
				}
			}
		}
	}()

	return out
}
