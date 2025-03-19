package ai

import (
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/workflow"
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
	tools  map[string]io.ReadWriteCloser
}

type AgentOption func(*Agent)

type AgentToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

/*
NewAgent creates a new agent with initialized components.
*/
func NewAgent(options ...AgentOption) *Agent {
	errnie.Debug("NewAgent")
	var browser *tools.Browser

	params := provider.NewParams(
		provider.WithModel(tweaker.GetModel(tweaker.GetProvider())),
		provider.WithTemperature(tweaker.GetTemperature()),
		provider.WithStream(tweaker.GetStream()),
	)

	agent := &Agent{
		buffer: stream.NewBuffer(func(evt *datura.Artifact) (err error) {
			errnie.Debug("agent.buffer.fn")
			var payload []byte

			if payload, err = evt.DecryptPayload(); err != nil {
				return errnie.Error(err)
			}

			if err = json.Unmarshal(payload, params); err != nil {
				return errnie.Error(err)
			}

			if len(params.Messages) < 3 {
				return
			}

			tc := &AgentToolCall{}

			if err = json.Unmarshal([]byte(
				params.Messages[len(params.Messages)-1].Content,
			), tc); err != nil {
				return errnie.Error(err)
			}

			args := map[string]any{}

			if err = json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				return errnie.Error(err)
			}

			if browser == nil {
				browser = tools.NewBrowser()
			}

			if err = evt.SetMetaValue("url", args["url"]); err != nil {
				return errnie.Error(err)
			}

			datura.WithPayload([]byte(args["script"].(string)))(evt)

			if _, err = io.Copy(browser, evt); err != nil {
				return errnie.Error(err)
			}

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

	agent.tools = map[string]io.ReadWriteCloser{
		"browser": workflow.NewFeedback(browser, agent),
	}

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

func WithModel(model string) AgentOption {
	return func(agent *Agent) {
		agent.params.Model = model
	}
}

func WithTools(tools []*provider.Tool) AgentOption {
	return func(agent *Agent) {
		agent.params.Tools = tools
	}
}

func WithProcess(proc provider.ResponseFormat) AgentOption {
	return func(agent *Agent) {
		agent.params.ResponseFormat = &provider.ResponseFormat{
			Name:        proc.Name,
			Description: proc.Description,
			Schema:      proc.Schema,
			Strict:      proc.Strict,
		}
	}
}

func WithSystem(system string) AgentOption {
	return func(agent *Agent) {
		if len(agent.params.Messages) == 0 {
			agent.params.Messages = append(agent.params.Messages, provider.NewMessage(
				provider.WithSystemRole(system),
			))

			return
		}

		agent.params.Messages[0].Content = system
	}
}

func WithUser(name, prompt string) AgentOption {
	return func(agent *Agent) {
		agent.params.Messages = append(agent.params.Messages, provider.NewMessage(
			provider.WithUserRole(name, prompt),
		))
	}
}
