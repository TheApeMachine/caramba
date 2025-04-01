package ai

import (
	"encoding/json"
	"fmt"
	"strings"

	"capnproto.org/go/capnp/v3"
	provider "github.com/theapemachine/caramba/pkg/api/provider"
	"github.com/theapemachine/caramba/pkg/api/tool"
	"github.com/theapemachine/caramba/pkg/errnie"
)

var Agents []*Agent

func NewCapnpAgent(name string) (*Agent, error) {
	errnie.Debug("ai.agent.NewCapnpAgent")

	var (
		arena = capnp.SingleSegment(nil)
		seg   *capnp.Segment
		agent Agent
		err   error
	)

	if _, seg, err = capnp.NewMessage(arena); err != nil {
		return nil, errnie.Error(err)
	}

	if agent, err = NewRootAgent(seg); err != nil {
		return nil, errnie.Error(err)
	}

	if err = agent.SetName(name); err != nil {
		return nil, errnie.Error(err)
	}

	providerParams, err := provider.NewProviderParams(seg)

	if err != nil {
		return nil, errnie.Error(err)
	}

	ml, err := provider.NewMessage_List(seg, 0)

	if err != nil {
		return nil, errnie.Error(err)
	}

	providerParams.SetMessages(ml)

	if err = providerParams.SetModel("gpt-4o-mini"); err != nil {
		return nil, errnie.Error(err)
	}

	if err = agent.SetContext(providerParams); err != nil {
		return nil, errnie.Error(err)
	}

	return &agent, nil
}

func (agent *Agent) AddMessage(role, name, content string) *provider.ProviderParams {
	errnie.Debug("ai.agent.AddMessage", "role", role, "name", name, "content", content)

	context, err := agent.Context()

	if err != nil {
		errnie.Error(err)
		return &context
	}

	message, err := provider.NewMessage(agent.Segment())

	if err != nil {
		errnie.Error(err)
		return &context
	}

	if err = message.SetRole(role); err != nil {
		errnie.Error(err)
		return &context
	}

	if role == "tool" {
		if err = message.SetReference(name); err != nil {
			errnie.Error(err)
			return &context
		}
	}

	if err = message.SetName(name); err != nil {
		errnie.Error(err)
		return &context
	}

	if err = message.SetContent(content); err != nil {
		errnie.Error(err)
		return &context
	}

	currentMessages, err := context.Messages()

	if err != nil {
		errnie.Error(err)
		return &context
	}

	messages, err := context.NewMessages(int32(currentMessages.Len() + 1))

	if err != nil {
		errnie.Error(err)
		return &context
	}

	for i := range currentMessages.Len() {
		if err := messages.Set(i, currentMessages.At(i)); err != nil {
			errnie.Error(err)
			return &context
		}
	}

	if err := messages.Set(currentMessages.Len(), message); err != nil {
		errnie.Error(err)
		return &context
	}

	if err := context.SetMessages(messages); err != nil {
		errnie.Error(err)
		return &context
	}

	return &context
}

func (agent *Agent) AddTool(toolName string) *provider.ProviderParams {
	errnie.Debug("ai.agent.AddTool")

	context, err := agent.Context()

	if err != nil {
		errnie.Error(err)
		return &context
	}

	newTool, err := tool.NewCapnpTool(toolName)

	if err != nil {
		errnie.Error(err)
		return &context
	}

	currentTools, err := context.Tools()

	if err != nil {
		errnie.Error(err)
		return &context
	}

	tools, err := context.NewTools(int32(currentTools.Len() + 1))

	if err != nil {
		errnie.Error(err)
		return &context
	}

	for i := range currentTools.Len() {
		if err := tools.Set(i, currentTools.At(i)); err != nil {
			errnie.Error(err)
			return &context
		}
	}

	if err := tools.Set(currentTools.Len(), *newTool); err != nil {
		errnie.Error(err)
		return &context
	}

	return &context
}

func (agent *Agent) Ask() *provider.ProviderParams {
	errnie.Debug("ai.agent.Ask")

	context, err := agent.Context()

	if err != nil {
		errnie.Error(err)
		return &context
	}

	prvdr := provider.NewProvider()
	prvdr.SetParams(&context)

	if err := prvdr.Generate(); err != nil {
		errnie.Error(err)
		return &context
	}

	agent.HandleToolCalls()

	return &context
}

func (agent *Agent) Send(
	agentName string,
	agents []*Agent,
	sysArgs map[string]any,
	toolId string,
) (err error) {
	if targetName, ok := sysArgs["to"].(string); ok {
		if msg, ok := sysArgs["message"].(string); ok {
			for _, targetAgent := range agents {
				targetAgentName, err := targetAgent.Name()

				if err != nil {
					continue
				}

				out := strings.Builder{}
				out.WriteString("MESSAGE RECEIVED\n\n")
				out.WriteString("FROM: " + agentName + "\n")
				out.WriteString("TO  : " + targetName + "\n")
				out.WriteString("\n" + msg + "\n")

				if targetAgentName == targetName {
					targetAgent.AddMessage(
						"user",
						agentName,
						out.String(),
					)

					fmt.Printf(
						"[%s -> %s]\n\n%s\n\n",
						agentName,
						targetName,
						out.String(),
					)

					agent.AddMessage(
						"tool",
						toolId,
						"Message sent to "+targetName,
					)

					break
				}
			}
		}
	}

	return nil
}

func (agent *Agent) Error(err error, toolId string) {
	errnie.Error(err)

	agent.AddMessage(
		"tool",
		toolId,
		err.Error(),
	)
}

func (agent *Agent) HandleToolCalls() *provider.ProviderParams {
	errnie.Debug("ai.agent.HandleToolCalls")

	context, err := agent.Context()

	if err != nil {
		errnie.Error(err)
		return &context
	}

	messages, err := context.Messages()

	if err != nil {
		errnie.Error(err)
		return &context
	}

	lastMessage := messages.At(messages.Len() - 1)

	if lastMessage.HasToolCalls() {
		toolCalls, err := lastMessage.ToolCalls()

		if err != nil {
			errnie.Error(err)
			return &context
		}

		for i := range toolCalls.Len() {
			toolCall := toolCalls.At(i)
			id, err := toolCall.Id()

			if err != nil {
				agent.Error(err, id)
				continue // Continue to next tool call instead of returning
			}

			function, err := toolCall.Function()

			if err != nil {
				agent.Error(err, id)
				continue // Continue to next tool call instead of returning
			}

			name, err := function.Name()

			if err != nil {
				agent.Error(err, id)
				continue // Continue to next tool call instead of returning
			}

			arguments, err := function.Arguments()

			if err != nil {
				agent.Error(err, id)
				continue // Continue to next tool call instead of returning
			}

			args := map[string]any{}

			if err := json.Unmarshal([]byte(arguments), &args); err != nil {
				agent.Error(err, id)
				continue // Continue to next tool call instead of returning
			}

			agentName, err := agent.Name()

			if err != nil {
				agent.Error(err, id)
				continue // Continue to next tool call instead of returning
			}

			// Track if a response was added for this tool call
			responseAdded := false

			switch name {
			case "inspect":
				if scope, ok := args["scope"].(string); ok {
					switch scope {
					case "agents":
						agent.InspectAgents(agentName, id)
						responseAdded = true
					default:
						// Handle unknown scope
						agent.Error(fmt.Errorf("unknown scope: %s", scope), id)
						responseAdded = true
					}
				} else {
					// Handle missing scope
					agent.Error(fmt.Errorf("missing scope parameter"), id)
					responseAdded = true
				}
			case "message":
				if to, ok := args["to"].(string); ok {
					if message, ok := args["message"].(string); ok {
						agent.Send(
							agentName,
							Agents,
							map[string]any{"to": to, "message": message},
							id,
						)
						responseAdded = true
					} else {
						agent.Error(fmt.Errorf("missing message parameter"), id)
						responseAdded = true
					}
				} else {
					agent.Error(fmt.Errorf("missing to parameter"), id)
					responseAdded = true
				}
			case "optimize":
				if _, ok := args["operation"].(string); ok {
					agent.Optimize(args, id)
					responseAdded = true
				} else {
					// Even if operation is missing, still add a response
					agent.AddMessage(
						"tool",
						id,
						"Operation completed with default settings",
					)
					responseAdded = true
				}
			default:
				agent.Error(fmt.Errorf("unknown tool call: %s", name), id)
				responseAdded = true
			}

			// If no response was added for this tool call, add a default one
			if !responseAdded {
				agent.AddMessage(
					"tool",
					id,
					"Tool call processed",
				)
			}
		}
	}

	return &context
}
