package ai

import (
	aiCtx "github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/event"
	"github.com/theapemachine/caramba/pkg/message"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

/*
Agent represents an entity that can process messages, interact with tools,
and produce responses. It implements io.ReadWriteCloser to enable composable
pipelines for complex workflows.
*/
type Agent struct {
	params *aiCtx.Artifact
	buffer *stream.Buffer
}

/*
NewAgent creates a new agent with initialized components.
*/
func NewAgent() *Agent {
	errnie.Debug("NewAgent")

	agent := &Agent{
		params: aiCtx.New(
			tweaker.GetModel(tweaker.GetProvider()),
			[]*aiCtx.Message{},
			[]*aiCtx.Tool{},
			[]byte{},
			tweaker.GetTemperature(),
			tweaker.GetTopP(),
			tweaker.GetTopK(),
			tweaker.GetPresencePenalty(),
			tweaker.GetFrequencyPenalty(),
			tweaker.GetMaxTokens(),
			tweaker.GetStream(),
		),
	}

	agent.buffer = stream.NewBuffer(
		func(evt *event.Artifact) (err error) {
			errnie.Debug("agent.buffer.fn", "event", evt)

			payload, err := evt.Payload()

			if errnie.Error(err) != nil {
				return err
			}

			msg := &message.Artifact{}
			_, err = msg.Write(payload)

			if errnie.Error(err) != nil {
				return err
			}

			err = agent.params.AddMessage(msg)
			if errnie.Error(err) != nil {
				return err
			}

			// Create a new event with the agent's params.
			newEvent := event.New(
				"agent",
				event.ContextEvent,
				event.UserRole,
				agent.params.Marshal(),
			)

			// Override the event with the new event.
			*evt = *newEvent

			return nil
		},
	)

	return agent
}

/*
Read implements io.Reader for Agent.

It reads from the internal context.
*/
func (agent *Agent) Read(p []byte) (n int, err error) {
	errnie.Debug("Agent.Read")
	return agent.buffer.Read(p)
}

/*
Write implements io.Writer for Agent.

It writes to the internal context.
*/
func (agent *Agent) Write(p []byte) (n int, err error) {
	errnie.Debug("Agent.Write")
	return agent.buffer.Write(p)
}

/*
Close implements io.Closer for Agent.

It closes the internal context.
*/
func (agent *Agent) Close() error {
	errnie.Debug("Agent.Close")
	return agent.params.Close()
}
