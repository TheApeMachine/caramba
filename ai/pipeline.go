package ai

import (
	"strings"
	"sync"

	"github.com/theapemachine/caramba/provider"
)

type Pipeline struct {
	stages []Stage
}

type Stage struct {
	agents     []*Agent
	parallel   bool
	aggregator func([]string) string
}

func NewPipeline() *Pipeline {
	return &Pipeline{}
}

func (p *Pipeline) AddStage(parallel bool, aggregator func([]string) string, agents ...*Agent) *Pipeline {
	p.stages = append(p.stages, Stage{
		agents:     agents,
		parallel:   parallel,
		aggregator: aggregator,
	})
	return p
}

// Execute runs the pipeline with the given input, orchestrating the flow between stages
func (p *Pipeline) Execute(input string) <-chan provider.Event {
	out := make(chan provider.Event)

	go func() {
		defer close(out)

		// Current context that gets passed between stages
		context := input

		// Process each stage
		for _, stage := range p.stages {
			var outputs []string

			if stage.parallel {
				outputs = p.executeParallel(stage, context, out)
			} else {
				outputs = p.executeSequential(stage, context, out)
			}

			// If there's an aggregator, use it to combine outputs for next stage
			if stage.aggregator != nil && len(outputs) > 0 {
				context = stage.aggregator(outputs)
			} else if len(outputs) > 0 {
				// If no aggregator, use the last output as context
				context = outputs[len(outputs)-1]
			}
		}

		out <- provider.Event{Type: provider.EventDone}
	}()

	return out
}

// executeParallel runs all agents in a stage concurrently
func (p *Pipeline) executeParallel(stage Stage, input string, out chan<- provider.Event) []string {
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
			for event := range agent.Generate(input) {
				// Forward all events to the output channel
				out <- event

				// Accumulate the response
				if event.Type == provider.EventToken {
					response.WriteString(event.Content)
				}
			}

			// Store the complete response
			mu.Lock()
			outputs[index] = response.String()
			mu.Unlock()
		}(i, agent)
	}

	wg.Wait()
	return outputs
}

// executeSequential runs agents in a stage one after another
func (p *Pipeline) executeSequential(stage Stage, input string, out chan<- provider.Event) []string {
	var outputs []string

	for _, agent := range stage.agents {
		var response strings.Builder

		for event := range agent.Generate(input) {
			// Forward all events to the output channel
			out <- event

			// Accumulate the response
			if event.Type == provider.EventToken {
				response.WriteString(event.Content)
			}
		}

		outputs = append(outputs, response.String())
		// Use the last response as input for the next agent
		input = response.String()
	}

	return outputs
}
