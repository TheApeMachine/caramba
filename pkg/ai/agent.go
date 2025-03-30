package ai

import (
	"fmt"
	"io"
	"time"

	"github.com/goombaio/namegenerator"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/workflow"
)

func init() {
	fmt.Println("ai.agent.init")
	provider.RegisterTool("agent")
}

func GenerateName() string {
	seed := time.Now().UTC().UnixNano()
	nameGenerator := namegenerator.NewNameGenerator(seed)

	name := nameGenerator.Generate()

	return name
}

/*
Agent represents an entity that can process messages, interact with tools,
and produce responses. It implements io.ReadWriteCloser to enable composable
pipelines for complex workflows.
*/
type Agent struct {
	Name   string
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
		Name:   GenerateName(),
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

	agent.buffer = stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
		errnie.Debug("agent.buffer.fn")

		switch artifact.Role() {
		case uint32(datura.ArtifactRoleInspect):
			switch artifact.Scope() {
			case uint32(datura.ArtifactScopeName):
				artifact.SetMetaValue("name", agent.Name)
			}

			return nil
		case uint32(datura.ArtifactRoleSignal):
			var payload []byte

			if payload, err = artifact.DecryptPayload(); err != nil {
				return errnie.Error(err)
			}

			// Add the response to the original artifact's Params payload.
			params.Messages = append(params.Messages, &provider.Message{
				Name:    agent.Name,
				Role:    provider.MessageRoleUser,
				Content: string(payload),
			})
		default:
			// Convert the artifact to a Params.
			newParams := provider.NewParams()
			if err = artifact.To(newParams); err != nil {
				return errnie.Error(err)
			}

			// Append the new messages to the existing ones
			params.Messages = append(params.Messages, newParams.Messages...)

			// Get the last message from the params.
			msg := params.Messages[len(params.Messages)-1]

			if len(msg.ToolCalls) == 0 {
				// No tool calls, so we can just stop here.
				errnie.Debug("no tool calls")
				return nil
			}

			// Iterate over the tool calls and copy them to the caller.
			for _, toolCall := range msg.ToolCalls {
				errnie.Debug("tool call", "function", toolCall.Function.Name, "arguments", toolCall.Function.Arguments)

				// Create an intermediary artifact to pass the tool call to the caller,
				// and receive the response.
				tc := datura.New(datura.WithPayload(toolCall.Marshal()))

				// Copy the tool call to the caller and receive the response.
				if err = workflow.NewFlipFlop(tc, agent.caller); err != nil {
					return errnie.Error(err)
				}

				// Add the response to the original artifact's Params payload.
				params.Messages = append(params.Messages, &provider.Message{
					Reference: toolCall.ID,
					Name:      agent.Name,
					Role:      provider.MessageRoleTool,
					Content:   datura.GetMetaValue[string](tc, "output"),
				})
			}

			datura.WithPayload(params.Marshal())(artifact)
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

func (agent *Agent) GetName() string {
	return agent.Name
}
