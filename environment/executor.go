package environment

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

type Executor struct {
	Agent       *ai.Agent
	chunks      chan *datura.Artifact
	accumulator *strings.Builder
}

func NewExecutor(agent *ai.Agent) *Executor {
	return &Executor{
		Agent:       agent,
		chunks:      make(chan *datura.Artifact, 100),
		accumulator: &strings.Builder{},
	}
}

func (executor *Executor) Run(ctx context.Context) {
	errnie.Info("⚡ "+executor.Agent.Identity.Name, "executor", "run")

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case chunk := <-executor.chunks:
				if chunk == nil {
					continue
				}
				executor.Agent.State = ai.AgentStateIterating
				decrypted, err := utils.DecryptPayload(chunk)

				if errnie.Error(err) != nil {
					executor.Agent.State = ai.AgentStateIdle
					continue
				}

				fmt.Print(string(decrypted))
				executor.accumulator.Write(decrypted)

			case msg := <-executor.Agent.Messages:
				errnie.Info("✉️ "+executor.Agent.Identity.Name, "executor", "message")

				if msg == nil {
					continue
				}
				executor.Agent.State = ai.AgentStateIterating

				go func() {
					defer func() {
						if executor.Agent.State != ai.AgentStateIterating {
							executor.Agent.State = ai.AgentStateIdle
						}
					}()

					executor.handleMessage(msg)

					for executor.Agent.State == ai.AgentStateIterating {
						streamChan := executor.Agent.Provider.Stream(executor.toArtifact())
						for chunk := range streamChan {
							executor.chunks <- chunk
						}

						if executor.accumulator.Len() > 0 {
							fmt.Println()
							executor.handleCompletion()
							executor.accumulator.Reset()
						}
					}
				}()

			default:
				time.Sleep(100 * time.Millisecond)
			}
		}
	}()
}

func (executor *Executor) handleMessage(msg *datura.Artifact) {
	errnie.Info("⚡ "+executor.Agent.Identity.Name, "executor", "handleMessage")

	if msg == nil {
		errnie.Error(fmt.Errorf("received nil message"))
		return
	}

	decrypted, err := utils.DecryptPayload(msg)

	if errnie.Error(err) != nil {
		return
	}

	// Add the message to agent's context
	executor.Agent.AddContext(string(decrypted))

	// Set state to iterating so the agent can process the message
	executor.Agent.State = ai.AgentStateIterating
}

func (executor *Executor) toArtifact() *datura.Artifact {
	errnie.Info("⚡ "+executor.Agent.Identity.Name, "context", "toArtifact")

	buf, err := json.Marshal(executor.Agent.Params)
	if errnie.Error(err) != nil {
		return nil
	}

	artifact := datura.NewArtifactBuilder(
		datura.MediaTypeApplicationJson,
		datura.ArtifactRoleAssistant,
		datura.ArtifactScopePrompt,
	)

	artifact.SetPayload(buf)

	out, err := artifact.Build()
	if errnie.Error(err) != nil {
		return nil
	}

	return out
}

func (executor *Executor) handleCompletion() {
	errnie.Info("⚡ "+executor.Agent.Identity.Name, "executor", "handleCompletion")

	executor.Agent.Params.Messages = append(executor.Agent.Params.Messages, provider.Message{
		Role:    "assistant",
		Content: executor.accumulator.String(),
	})

	// Extract JSON objects from the accumulated content
	toolCalls := utils.ExtractJSON(executor.accumulator.String())

	for _, toolCall := range toolCalls {
		artifact := datura.NewArtifactBuilder(
			datura.MediaTypeApplicationJson,
			datura.ArtifactRoleAssistant,
			datura.ArtifactScope(datura.ArtifactRoleUnknown),
		)

		payload, err := json.Marshal(toolCall)
		if errnie.Error(err) != nil {
			continue
		}

		artifact.SetPayload(payload)

		out, err := artifact.Build()
		if errnie.Error(err) != nil {
			continue
		}

		if toolName, ok := toolCall["name"].(string); ok {
			switch toolName {
			case "completion":
				tool := tools.NewCompletionTool()
				tool.Use(executor.Agent, out)
			case "message":
				tool := tools.NewMessageTool()
				tool.Use(executor.Agent, out)
				// Keep iterating after sending a message
				executor.Agent.State = ai.AgentStateIterating
			case "agent":
				tool := tools.NewAgentTool()
				tool.Use(executor.Agent, out)

				for _, agent := range executor.Agent.Agents {
					for _, delegate := range agent {
						if NewPool().GetExecutor(delegate.Identity.ID) == nil {
							NewPool().AddExecutor(NewExecutor(delegate))
						}
					}
				}
			}
		}
	}
}
