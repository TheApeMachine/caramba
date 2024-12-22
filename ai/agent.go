package ai

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/log"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
)

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
}

func NewAgent(role string, process Process) *Agent {
	agent := &Agent{
		provider:    provider.NewBalancedProvider(),
		name:        utils.NewName(),
		role:        role,
		process:     process,
		prompt:      NewPrompt(role, process),
		buffer:      NewBuffer(),
		temperature: 0.0,
		topP:        0.0,
		topK:        0,
	}

	// Set system prompt once during initialization
	agent.buffer.Poke(provider.Message{
		Role:    "system",
		Content: agent.prompt.Build(),
	})

	return agent
}

func (agent *Agent) Generate(prompt string) <-chan provider.Event {
	log.Info("generating", "agent", agent.name, "role", agent.role)

	out := make(chan provider.Event)

	go func() {
		defer close(out)

		agent.buffer.Poke(provider.Message{
			Role:    "user",
			Content: prompt,
		})

		if _, ok := agent.process.(Tool); ok {
			agent.process.(Tool).Initialize()
		}

		for {
			var response strings.Builder
			for event := range agent.provider.Generate(provider.GenerationParams{
				Messages:    agent.buffer.Peek(),
				Temperature: agent.temperature,
				TopP:        agent.topP,
				TopK:        agent.topK,
			}) {
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
					Role:    "assistant",
					Content: toolResponse,
				})

				out <- provider.Event{
					Type:    provider.EventFunctionCall,
					Content: toolResponse,
				}
			}

			if strings.Contains(response.String(), "<task-complete>") {
				break
			}
		}
	}()

	return out
}

func (agent *Agent) toolCall(response string) string {
	blocks := utils.ExtractJSONBlocks(response)
	log.Info("Extracted JSON blocks", "blocks", blocks)

	var output strings.Builder

	for _, block := range blocks {
		tool := agent.process.(Tool)
		log.Info("Calling tool with block", "block", block)
		output.WriteString(tool.Use(block))
	}

	return output.String()
}
