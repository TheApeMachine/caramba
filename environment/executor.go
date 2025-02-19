package environment

import (
	"context"
	"encoding/json"
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
		chunks:      make(chan *datura.Artifact),
		accumulator: &strings.Builder{},
	}
}

func (executor *Executor) Run(ctx context.Context) {
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case chunk := <-executor.chunks:
				executor.Agent.State = ai.AgentStateProcessing
				decrypted, err := utils.DecryptPayload(chunk)

				if errnie.Error(err) != nil {
					continue
				}

				executor.accumulator.Write(decrypted)
			case msg := <-executor.Agent.Messages:
				executor.Agent.State = ai.AgentStateProcessing

				go func() {
					for executor.Agent.State == ai.AgentStateProcessing {
						executor.chunks <- <-executor.Agent.Provider.Stream(
							executor.handleMessage(msg),
						)

						if executor.accumulator.Len() > 0 {
							executor.handleCompletion()
						}

						executor.accumulator.Reset()
					}
				}()
			default:
				time.Sleep(100 * time.Millisecond)
				executor.Agent.State = ai.AgentStateIdle
			}
		}
	}()
}

func (executor *Executor) handleMessage(msg *datura.Artifact) *datura.Artifact {
	decrypted, err := utils.DecryptPayload(msg)

	if errnie.Error(err) != nil {
		return nil
	}

	executor.Agent.AddContext(string(decrypted))

	buf, err := json.Marshal(executor.Agent.Params)

	if errnie.Error(err) != nil {
		return nil
	}

	artifact := datura.NewArtifactBuilder(
		datura.MediaTypeApplicationJson,
		datura.ArtifactRoleAssistant,
		datura.ArtifactScope(datura.ArtifactRoleUnknown),
	)

	artifact.SetPayload(buf)

	out, err := artifact.Build()

	if errnie.Error(err) != nil {
		return nil
	}

	return out
}

func (executor *Executor) handleCompletion() {
	executor.Agent.Params.Messages = append(executor.Agent.Params.Messages, provider.Message{
		Role:    "assistant",
		Content: executor.accumulator.String(),
	})

	toolcalls := make(map[string]any)

	err := json.Unmarshal([]byte(executor.accumulator.String()), &toolcalls)

	if errnie.Error(err) != nil {
		return
	}

	artifact := datura.NewArtifactBuilder(
		datura.MediaTypeApplicationJson,
		datura.ArtifactRoleAssistant,
		datura.ArtifactScope(datura.ArtifactRoleUnknown),
	)

	artifact.SetPayload([]byte(executor.accumulator.String()))

	out, err := artifact.Build()

	if errnie.Error(err) != nil {
		return
	}

	if toolName, ok := toolcalls["name"].(string); ok {
		switch toolName {
		case "completion":
			tool := tools.NewCompletionTool()
			tool.Use(executor.Agent, out)
		case "message":
			tool := tools.NewMessageTool()
			tool.Use(executor.Agent, out)
		}
	}
}
