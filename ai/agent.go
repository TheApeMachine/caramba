package ai

import (
	"strings"

	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

/*
Agent is a wrapper around a provider, which can be used to generate text.
An agent has either a process, or a tool. A tool is also a process, but
a process is not a tool. A process is just a jsonschema that is used by
the model to respond with a (partially) structured response. A tool
interfaces with anything that gives the agent additional functionality.
*/
type Agent struct {
	provider    provider.Provider
	name        string
	role        string
	process     Process
	prompt      *Prompt
	buffer      *Buffer
	temperature float64
	topP        float64
	topK        int
	maxIter     int
}

/*
NewAgent creates a new Agent instance.
*/
func NewAgent(role string, process Process, maxIter int) *Agent {
	name := utils.NewName()

	agent := &Agent{
		provider:    provider.NewBalancedProvider(),
		name:        name,
		role:        role,
		process:     process,
		prompt:      NewPrompt(name, role, process),
		buffer:      NewBuffer(),
		temperature: 0.0,
		topP:        0.0,
		topK:        0,
		maxIter:     maxIter,
	}

	agent.buffer.Poke(provider.Message{
		Role:    "system",
		Content: agent.prompt.Build(),
	})

	return agent
}

/*
Generate calls the underlying provider to have a Large Language Model
generate text for the agent.
*/
func (agent *Agent) Generate() <-chan provider.Event {
	out := make(chan provider.Event)

	go func() {
		defer close(out)

		iteration := 0

		for {
			if iteration >= agent.maxIter {
				break
			}

			agent.buffer.Poke(provider.Message{
				Role:    "assistant",
				Content: agent.prompt.BuildStatus(iteration, agent.maxIter-iteration),
			})

			var response strings.Builder

			params := provider.GenerationParams{
				Messages:    agent.buffer.Peek(),
				Temperature: agent.temperature,
				TopP:        agent.topP,
				TopK:        agent.topK,
			}

			for event := range agent.provider.Generate(params) {
				response.WriteString(event.Content)
				out <- event
			}

			agent.buffer.Poke(provider.Message{
				Role:    "assistant",
				Content: response.String(),
			})

			if _, ok := agent.process.(Tool); ok {
				toolResponse := agent.toolCall(response.String())

				agent.buffer.Poke(provider.Message{
					Role: "assistant",
					Content: utils.QuickWrap(
						"tool", toolResponse,
					),
				})

				out <- provider.Event{
					Type:    provider.EventToken,
					Content: toolResponse,
				}
			}

			iteration++
			errnie.Log("\n\n\n===AGENT===\n%s\n\n\n===========", agent.buffer.Peek())

			if strings.Contains(response.String(), "<task-complete>") {
				out <- provider.Event{
					Type:    provider.EventDone,
					Content: "\n",
				}
				break
			}
		}
	}()

	return out
}

func (agent *Agent) toolCall(response string) string {
	blocks := utils.ExtractJSONBlocks(response)

	var output strings.Builder

	for _, block := range blocks {
		tool := agent.process.(Tool)
		output.WriteString(tool.Use(block))
	}

	return output.String()
}
