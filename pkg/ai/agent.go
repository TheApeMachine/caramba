package ai

import (
	aiCtx "github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

/*
Agent represents an entity that can process messages, interact with tools,
and produce responses. It implements io.ReadWriteCloser to enable composable
pipelines for complex workflows.
*/
type Agent struct {
	buffer *stream.Buffer
	params *aiCtx.Artifact
}

/*
NewAgent creates a new agent with initialized components.
*/
func NewAgent() *Agent {
	errnie.Debug("NewAgent")

	return &Agent{
		buffer: stream.NewBuffer(func(evt *datura.Artifact) (err error) {
			errnie.Debug("agent.buffer.fn", "event", evt)
			return nil
		}),
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
