package ai

import (
	"encoding/json"
	"io"

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
	caller io.ReadWriteCloser
}

type AgentOption func(*Agent)

/*
NewAgent creates a new agent with initialized components.
*/
func NewAgent(options ...AgentOption) *Agent {
	errnie.Debug("NewAgent")

	params := provider.NewParams(
		provider.WithModel(tweaker.GetModel(tweaker.GetProvider())),
		provider.WithTemperature(tweaker.GetTemperature()),
		provider.WithStream(tweaker.GetStream()),
	)

	agent := &Agent{
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

	agent.buffer = stream.NewBuffer(func(evt *datura.Artifact) (err error) {
		errnie.Debug("agent.buffer.fn")
		var payload []byte

		if payload, err = evt.DecryptPayload(); err != nil {
			return errnie.Error(err)
		}

		if err = json.Unmarshal(payload, params); err != nil {
			return errnie.Error(err)
		}

		msg := params.Messages[len(params.Messages)-1]

		switch msg.Role {
		case provider.MessageRoleTool:
			toolcall := datura.New(
				datura.WithPayload([]byte(msg.Content)),
			)

			if _, err = io.Copy(agent.caller, toolcall); err != nil {
				return errnie.Error(err)
			}

			if _, err = io.Copy(toolcall, agent.caller); err != nil {
				return errnie.Error(err)
			}
		}

		return nil
	})

	for _, option := range options {
		option(agent)
	}

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
	return agent.buffer.Close()
}

func WithCaller(caller io.ReadWriteCloser) AgentOption {
	return func(agent *Agent) {
		agent.caller = caller
	}
}
