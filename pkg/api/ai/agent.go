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

	// Copy existing messages
	for i := range currentMessages.Len() {
		if err := messages.Set(i, currentMessages.At(i)); err != nil {
			errnie.Error(err)
			return &context
		}
	}

	// Add new message
	if err := messages.Set(currentMessages.Len(), message); err != nil {
		errnie.Error(err)
		return &context
	}

	// Set the updated messages list back to the context
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

	// Copy existing tools
	for i := range currentTools.Len() {
		if err := tools.Set(i, currentTools.At(i)); err != nil {
			errnie.Error(err)
			return &context
		}
	}

	// Add new tool
	if err := tools.Set(currentTools.Len(), *newTool); err != nil {
		errnie.Error(err)
		return &context
	}

	return &context
}

// Ask sends a request to the agent and returns the updated context
func (agent *Agent) Ask() *provider.ProviderParams {
	errnie.Debug("ai.agent.Ask")

	context, err := agent.Context()

	if err != nil {
		errnie.Error(err)
		return &context
	}

	prvdr := provider.NewProvider()
	prvdr.SetParams(&context)

	// Call the provider's Generate method which will update the ProviderParams directly
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
	if targetName, ok := sysArgs["send_to_arg"].(string); ok {
		if msg, ok := sysArgs["message_arg"].(string); ok {
			// Find the target agent
			for _, targetAgent := range agents {
				targetAgentName, err := targetAgent.Name()
				if err != nil {
					continue
				}

				out := strings.Builder{}
				out.WriteString("MESSAGE RECEIVED\n")
				out.WriteString("FROM: " + agentName + "\n")
				out.WriteString("TO  : " + targetName + "\n")
				out.WriteString("\n" + msg + "\n")

				if targetAgentName == targetName {
					// Add the message to the target agent
					targetAgent.AddMessage(
						"user",
						agentName,
						out.String(),
					)

					fmt.Printf("[%s -> %s]\n\n%s\n\n", agentName, targetName, out.String())

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

			function, err := toolCall.Function()

			if err != nil {
				errnie.Error(err)
				return &context
			}

			name, err := function.Name()

			if err != nil {
				errnie.Error(err)
				return &context
			}

			arguments, err := function.Arguments()

			if err != nil {
				errnie.Error(err)
				return &context
			}

			args := map[string]any{}

			if err := json.Unmarshal([]byte(arguments), &args); err != nil {
				errnie.Error(err)
				return &context
			}

			agentName, err := agent.Name()

			if err != nil {
				errnie.Error(err)
				return &context
			}

			id, err := toolCall.Id()

			if err != nil {
				errnie.Error(err)
				return &context
			}

			switch name {
			case "system":
				if command, ok := args["command"].(string); ok {
					switch command {
					case "inspect":
						out := strings.Builder{}

						out.WriteString("INSPECTING SYSTEM\n")
						for _, targetAgent := range Agents {
							targetName, err := targetAgent.Name()

							if err != nil {
								errnie.Error(err)
								continue
							}

							out.WriteString(targetName + " (agent)\n")
						}

						agent.AddMessage(
							"tool",
							id,
							out.String(),
						)

						fmt.Printf("[%s]\n\n%s\n\n", agentName, out.String())
					case "send":
						agent.Send(
							agentName,
							Agents,
							args,
							id,
						)
					}
				}
			}
		}
	}

	return &context
}
