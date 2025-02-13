package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/davecgh/go-spew/spew"
	sdk "github.com/openai/openai-go"

	"github.com/theapemachine/caramba/process/persona"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/system"
	"github.com/theapemachine/errnie"
)

type Agent struct {
	ID       string
	Name     string
	ctx      context.Context
	cancel   context.CancelFunc
	provider *provider.OpenAI
	i        chan *provider.StructuredParams
	o        chan *system.Envelope
}

func NewAgent(id, name string) *Agent {
	i, o := system.NewQueue().Claim(id)

	return &Agent{
		ID:       id,
		Name:     name,
		provider: provider.NewOpenAI(os.Getenv("OPENAI_API_KEY")),
		i:        o,
		o:        i,
	}
}

func (agent *Agent) Stream() {
	errnie.Info("streaming", "agent", agent.Name)
	agent.ctx, agent.cancel = context.WithCancel(context.Background())

	go func() {
		for {
			select {
			case <-agent.ctx.Done():
				return
			case message := <-agent.i:
				errnie.Info("receive", "agent", agent.Name)
				completion, err := agent.provider.Stream(message)

				if err != nil {
					message.Messages = append(
						message.Messages, sdk.AssistantMessage(err.Error()),
					)

					continue
				}

				agent.handleCompletion(message, completion)
			default:
				time.Sleep(1 * time.Second)
			}
		}
	}()
}

func (agent *Agent) handleCompletion(
	message *provider.StructuredParams, completion *sdk.ChatCompletion,
) {
	errnie.Info("handling completion", "agent", agent.Name)

	if len(completion.Choices) == 0 {
		return
	}

	if completion.Choices[0].Message.Content != "" {
		content := completion.Choices[0].Message.Content
		fmt.Println(content)

		message.Messages = append(
			message.Messages,
			sdk.AssistantMessage(content),
		)

		agent.handleCommands(message, content)
	}

	if completion.Choices[0].Message.ToolCalls != nil {
		for _, toolCall := range completion.Choices[0].Message.ToolCalls {
			message.Messages = append(
				message.Messages, completion.Choices[0].Message,
			)
			agent.handleToolCall(message, toolCall)
		}
	}
}

func (agent *Agent) handleCommands(
	message *provider.StructuredParams, content string,
) {
	buf := &persona.Agent{}
	json.Unmarshal([]byte(content), buf)

	switch buf.Inspect {
	case "system":
		agent.inspectSystem()
	case "agents":
		agent.inspectAgents()
	case "topics":
		agent.inspectTopics()
	}
}

func (agent *Agent) handleToolCall(
	message *provider.StructuredParams, toolCall sdk.ChatCompletionMessageToolCall,
) {
	errnie.Info("handling toolcall", "agent", agent.Name)

	message.Messages = append(message.Messages, sdk.ToolMessage(
		toolCall.ID, "tool not implemented",
	))
}
