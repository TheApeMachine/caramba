package ai

import (
	"strconv"
	"strings"
	"sync"

	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
)

type Pipeline struct {
	stages []Stage
}

type Stage struct {
	agents     []*Agent
	parallel   bool
	aggregator func([]string) string
}

// Add this new type to represent combined context
type StageContext struct {
	originalPrompt string
	currentContext string
}

func NewPipeline() *Pipeline {
	return &Pipeline{}
}

func (p *Pipeline) AddSequentialStage(
	aggregator func([]string) string,
	agents ...*Agent,
) *Pipeline {
	p.stages = append(p.stages, Stage{
		agents:     agents,
		parallel:   false,
		aggregator: aggregator,
	})
	return p
}

func (pipeline *Pipeline) AddParallelStage(
	aggregator func([]string) string,
	agents ...*Agent,
) *Pipeline {
	pipeline.stages = append(pipeline.stages, Stage{
		agents:     agents,
		parallel:   true,
		aggregator: aggregator,
	})
	return pipeline
}

/*
Execute the pipeline with the given input, orchestrating the flow between stages
*/
func (p *Pipeline) Execute(input string) <-chan provider.Event {
	out := make(chan provider.Event)

	utils.JoinWith("\n",
		"<user-prompt>",
		input,
		"</user-prompt>",
	)

	go func() {
		defer close(out)

		// Initialize context with both original prompt and current context
		ctx := StageContext{
			originalPrompt: input,
			currentContext: input,
		}

		// Process each stage
		for _, stage := range p.stages {
			var outputs []string

			if stage.parallel {
				outputs = p.executeParallel(stage, ctx, out)
			} else {
				outputs = p.executeSequential(stage, ctx, out)
			}

			// If there's an aggregator, use it to combine outputs for next stage
			if stage.aggregator != nil && len(outputs) > 0 {
				ctx.currentContext = stage.aggregator(outputs)
			} else if len(outputs) > 0 {
				// If no aggregator, use the last output as context
				ctx.currentContext = outputs[len(outputs)-1]
			}
		}
	}()

	return out
}

// executeParallel runs all agents in a stage concurrently
func (p *Pipeline) executeParallel(stage Stage, ctx StageContext, out chan<- provider.Event) []string {
	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		outputs = make([]string, len(stage.agents))
	)

	for i, agent := range stage.agents {
		wg.Add(1)
		go func(index int, agent *Agent) {
			defer wg.Done()

			var response strings.Builder

			// Wrap context for each parallel agent
			agentContext := utils.JoinWith("\n",
				"<parallel_context agent="+agent.name+" index="+strconv.Itoa(index)+">",
				ctx.currentContext,
				"</parallel_context>",
			)

			for event := range agent.Generate(agentContext) {
				out <- event
				if event.Type == provider.EventToken {
					response.WriteString(event.Content)
				}
			}

			mu.Lock()
			outputs[index] = response.String()
			mu.Unlock()
		}(i, agent)
	}

	wg.Wait()
	return outputs
}

// executeSequential runs agents in a stage one after another
func (p *Pipeline) executeSequential(stage Stage, ctx StageContext, out chan<- provider.Event) []string {
	var outputs []string
	currentInput := ctx.currentContext

	for i, agent := range stage.agents {
		var response strings.Builder

		// Wrap context for each sequential agent
		agentContext := utils.JoinWith("\n",
			"<sequential_context agent="+agent.name+" index="+strconv.Itoa(i)+">",
			currentInput,
			"</sequential_context>",
		)

		for event := range agent.Generate(agentContext) {
			out <- event
			if event.Type == provider.EventToken {
				response.WriteString(event.Content)
			}
		}

		outputs = append(outputs, response.String())
		currentInput = response.String()
	}

	return outputs
}
