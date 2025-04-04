package agent

import (
	"capnproto.org/go/capnp/v3"
	"github.com/google/uuid"
	aicontext "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools"
)

// AgentOption defines a function that configures an Agent
type AgentOption func(*Agent) error

// New creates a new agent with the provided options
func New(options ...AgentOption) *Agent {
	var (
		arena = capnp.SingleSegment(nil)
		seg   *capnp.Segment
		agent Agent
		err   error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil
	}

	if agent, err = NewRootAgent(seg); errnie.Error(err) != nil {
		return nil
	}

	// Create and set identity with default ID
	identity, err := agent.NewIdentity()
	if errnie.Error(err) != nil {
		return nil
	}

	if errnie.Error(identity.SetIdentifier(uuid.New().String())) != nil {
		return nil
	}

	// Apply all options
	for _, option := range options {
		if err := option(&agent); errnie.Error(err) != nil {
			return nil
		}
	}

	return &agent
}

// WithName sets the agent's name
func WithName(name string) AgentOption {
	return func(a *Agent) error {
		identity, err := a.Identity()
		if err != nil {
			return errnie.Error(err)
		}
		return errnie.Error(identity.SetName(name))
	}
}

// WithRole sets the agent's role
func WithRole(role string) AgentOption {
	return func(a *Agent) error {
		identity, err := a.Identity()
		if err != nil {
			return errnie.Error(err)
		}
		return errnie.Error(identity.SetRole(role))
	}
}

// WithIdentifier sets the agent's identifier
func WithIdentifier(id string) AgentOption {
	return func(a *Agent) error {
		identity, err := a.Identity()
		if err != nil {
			return errnie.Error(err)
		}
		return errnie.Error(identity.SetIdentifier(id))
	}
}

// WithModel sets the model to use for the agent
func WithModel(model string) AgentOption {
	return func(a *Agent) error {
		params, err := a.Params()
		if err != nil {
			params, err = a.NewParams()
			if err != nil {
				return errnie.Error(err)
			}
		}
		return errnie.Error(params.SetModel(model))
	}
}

// WithSystemMessage adds a system message to the agent's context
func WithSystemMessage(content string) AgentOption {
	return func(a *Agent) error {
		ctx, err := a.Context()
		if err != nil {
			ctx, err = a.NewContext()
			if err != nil {
				return errnie.Error(err)
			}
		}

		// Get existing messages or create new ones
		var messages aicontext.Message_List
		if ctx.HasMessages() {
			messages, err = ctx.Messages()
			if err != nil {
				return errnie.Error(err)
			}

			// Create new list with one more message
			newMessages, err := ctx.NewMessages(int32(messages.Len() + 1))
			if err != nil {
				return errnie.Error(err)
			}

			// Copy existing messages
			for i := 0; i < messages.Len(); i++ {
				newMessages.Set(i, messages.At(i))
			}

			// Add new system message
			sysMsg := newMessages.At(messages.Len())
			if err := sysMsg.SetRole("system"); err != nil {
				return errnie.Error(err)
			}
			if err := sysMsg.SetContent(content); err != nil {
				return errnie.Error(err)
			}

			messages = newMessages
		} else {
			// Create new messages list with just one system message
			messages, err = ctx.NewMessages(1)
			if err != nil {
				return errnie.Error(err)
			}

			sysMsg := messages.At(0)
			if err := sysMsg.SetRole("system"); err != nil {
				return errnie.Error(err)
			}
			if err := sysMsg.SetContent(content); err != nil {
				return errnie.Error(err)
			}
		}

		return nil
	}
}

// WithUserMessage adds a user message to the agent's context
func WithUserMessage(content string) AgentOption {
	return func(a *Agent) error {
		ctx, err := a.Context()
		if err != nil {
			ctx, err = a.NewContext()
			if err != nil {
				return errnie.Error(err)
			}
		}

		// Get existing messages or create new ones
		var messages aicontext.Message_List
		if ctx.HasMessages() {
			messages, err = ctx.Messages()
			if err != nil {
				return errnie.Error(err)
			}

			// Create new list with one more message
			newMessages, err := ctx.NewMessages(int32(messages.Len() + 1))
			if err != nil {
				return errnie.Error(err)
			}

			// Copy existing messages
			for i := 0; i < messages.Len(); i++ {
				newMessages.Set(i, messages.At(i))
			}

			// Add new user message
			userMsg := newMessages.At(messages.Len())
			if err := userMsg.SetRole("user"); err != nil {
				return errnie.Error(err)
			}
			if err := userMsg.SetContent(content); err != nil {
				return errnie.Error(err)
			}

			messages = newMessages
		} else {
			// Create new messages list with just one user message
			messages, err = ctx.NewMessages(1)
			if err != nil {
				return errnie.Error(err)
			}

			userMsg := messages.At(0)
			if err := userMsg.SetRole("user"); err != nil {
				return errnie.Error(err)
			}
			if err := userMsg.SetContent(content); err != nil {
				return errnie.Error(err)
			}
		}

		return nil
	}
}

// WithTool adds a tool to the agent
func WithTool(name, description string, properties map[string]ToolProperty) AgentOption {
	return func(a *Agent) error {
		var toolsList tools.Tool_List

		if a.HasTools() {
			var err error
			toolsList, err = a.Tools()
			if err != nil {
				return errnie.Error(err)
			}

			// Create a new list with one more tool
			newToolsList, err := a.NewTools(int32(toolsList.Len() + 1))
			if err != nil {
				return errnie.Error(err)
			}

			// Copy existing tools
			for i := 0; i < toolsList.Len(); i++ {
				newToolsList.Set(i, toolsList.At(i))
			}

			// Add new tool
			if err := addTool(newToolsList.At(toolsList.Len()), name, description, properties); err != nil {
				return err
			}

			toolsList = newToolsList
		} else {
			var err error
			toolsList, err = a.NewTools(1)
			if err != nil {
				return errnie.Error(err)
			}

			if err := addTool(toolsList.At(0), name, description, properties); err != nil {
				return err
			}
		}

		return nil
	}
}

// ToolProperty defines a property for a tool parameter
type ToolProperty struct {
	Type        string
	Description string
	Required    bool
	EnumValues  []string
}

// Helper function to add a tool to a tool list
func addTool(tool tools.Tool, name, description string, properties map[string]ToolProperty) error {
	function, err := tool.NewFunction()
	if err != nil {
		return errnie.Error(err)
	}

	if err := function.SetName(name); err != nil {
		return errnie.Error(err)
	}

	if err := function.SetDescription(description); err != nil {
		return errnie.Error(err)
	}

	// Create parameters
	params, err := function.NewParameters()
	if err != nil {
		return errnie.Error(err)
	}

	// Count required properties
	requiredCount := 0
	for _, prop := range properties {
		if prop.Required {
			requiredCount++
		}
	}

	// Create properties list
	propList, err := params.NewProperties(int32(len(properties)))
	if err != nil {
		return errnie.Error(err)
	}

	// Create required list
	reqList, err := params.NewRequired(int32(requiredCount))
	if err != nil {
		return errnie.Error(err)
	}

	// Set properties and required fields
	i := 0
	reqIdx := 0
	for name, prop := range properties {
		property := propList.At(i)

		if err := property.SetName(name); err != nil {
			return errnie.Error(err)
		}

		if err := property.SetType(prop.Type); err != nil {
			return errnie.Error(err)
		}

		if err := property.SetDescription(prop.Description); err != nil {
			return errnie.Error(err)
		}

		// Set enum values if provided
		if len(prop.EnumValues) > 0 {
			enumList, err := property.NewEnum(int32(len(prop.EnumValues)))
			if err != nil {
				return errnie.Error(err)
			}

			for j, val := range prop.EnumValues {
				if err := enumList.Set(j, val); err != nil {
					return errnie.Error(err)
				}
			}
		}

		// Add to required list if needed
		if prop.Required {
			if err := reqList.Set(reqIdx, name); err != nil {
				return errnie.Error(err)
			}
			reqIdx++
		}

		i++
	}

	return nil
}
