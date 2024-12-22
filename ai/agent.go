package ai

import (
	"os"
	"strconv"
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
	maxIter     int
}

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

		// Wrap incoming prompt with context metadata
		agent.buffer.Poke(provider.Message{
			Role: "user",
			Content: utils.JoinWith("\n",
				"<input_context agent="+agent.name+" role="+agent.role+">",
				prompt,
				"</input_context>",
			),
		})

		if _, ok := agent.process.(Tool); ok {
			if err := agent.process.(Tool).Initialize(); err != nil {
				log.Error("Failed to initialize tool", "error", err)
				os.Exit(1)
			}
		}

		iteration := 0

		for {
			if iteration >= agent.maxIter {
				out <- provider.Event{
					Type:    provider.EventDone,
					Content: "\n",
				}
				break
			}

			var response strings.Builder
			// Add task wrapper before generation
			currentMessages := agent.buffer.Peek()
			currentMessages = append(currentMessages, provider.Message{
				Role: "system",
				Content: utils.JoinWith("\n",
					"<task>",
					"  Based on the provided context:",
					"    1. Consider only information relevant to your role as "+agent.role,
					"    2. Focus on your specific responsibility in the current stage",
					"    3. Generate exactly one meaningful step towards the goal",
					"    4. Respond according to the required protocol",
					"    5. Do not let the context distract or confuse you, use only what is relevant to you right now",
					"",
					"  <iteration_status>",
					"    Current iteration: "+strconv.Itoa(iteration),
					"    Remaining iterations: "+strconv.Itoa(agent.maxIter-iteration),
					"    Complete task when either:",
					"      - No more meaningful steps are needed",
					"      - Reached maximum iterations ("+strconv.Itoa(agent.maxIter)+")",
					"  </iteration_status>",
					"</task>",
				),
			})

			for event := range agent.provider.Generate(provider.GenerationParams{
				Messages:    currentMessages,
				Temperature: agent.temperature,
				TopP:        agent.topP,
				TopK:        agent.topK,
			}) {
				response.WriteString(event.Content)
				out <- event
			}

			for _, ignore := range []string{"apologize", "sorry", "apologies"} {
				if strings.Contains(response.String(), ignore) {
					if strings.Contains(response.String(), "<task-complete>") {
						out <- provider.Event{
							Type:    provider.EventDone,
							Content: "\n",
						}
						return
					}
				}
			}

			agent.buffer.Poke(provider.Message{
				Role: "assistant",
				Content: utils.JoinWith("\n",
					"<agent_response",
					"  name=\""+agent.name+"\"",
					"  role=\""+agent.role+"\"",
					"  iteration=\""+strconv.Itoa(iteration)+"\"",
					">",
					response.String(),
					"</agent_response>",
				),
			})

			if _, ok := agent.process.(Tool); ok {
				toolResponse := agent.toolCall(response.String())

				agent.buffer.Poke(provider.Message{
					Role: "assistant",
					Content: utils.JoinWith("\n",
						"<tool_response",
						"  agent=\""+agent.name+"\"",
						"  role=\""+agent.role+"\"",
						"  iteration=\""+strconv.Itoa(iteration)+"\"",
						">",
						toolResponse,
						"</tool_response>",
					),
				})

				out <- provider.Event{
					Type:    provider.EventFunctionCall,
					Content: toolResponse,
				}
			}

			iteration++

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
