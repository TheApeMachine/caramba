package ai

import (
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

func init() {
	provider.RegisterTool("agent")
}

/*
Agent represents an entity that can process messages, interact with tools,
and produce responses. It implements io.ReadWriteCloser to enable composable
pipelines for complex workflows.
*/
type Agent struct {
	buffer *stream.Buffer
	params *provider.Params
	Schema *provider.Tool
}

/*
NewAgent creates a new agent with initialized components.
*/
func NewAgent() *Agent {
	errnie.Debug("NewAgent")

	params := provider.NewParams(
		provider.WithModel(tweaker.GetModel(tweaker.GetProvider())),
		provider.WithTemperature(tweaker.GetTemperature()),
		provider.WithStream(tweaker.GetStream()),
	)

	return &Agent{
		buffer: stream.NewBuffer(func(evt *datura.Artifact) (err error) {
			errnie.Debug("agent.buffer.fn")
			var payload []byte

			if payload, err = evt.EncryptedPayload(); err != nil {
				return errnie.Error(err)
			}

			params.Unmarshal(payload)
			return nil
		}),
		params: params,
		Schema: provider.NewTool(
			provider.WithFunction(
				"agent",
				"An agent that can process messages, interact with tools, and produce responses.",
			),
			provider.WithProperty(
				"model",
				"string",
				"The model to use for the agent.",
				[]any{"gpt-4o-mini", "gemini-2.0-flash-thinking"},
			),
			provider.WithProperty(
				"tools",
				"array",
				"The tools the agent should have access to.",
				provider.RegisteredTools,
			),
			provider.WithProperty(
				"system",
				"string",
				"The system message to use for the agent.",
				[]any{},
			),
			provider.WithProperty(
				"user",
				"string",
				"The user message to use for the agent.",
				[]any{},
			),
			provider.WithProperty(
				"temperature",
				"number",
				"The temperature to use for the agent.",
				[]any{},
			),
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
	return agent.buffer.Close()
}
