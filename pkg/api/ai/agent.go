package ai

import (
	"context"
	"fmt"

	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/api/provider"
)

// CapnpAgent implements the Cap'n Proto Agent interface
type CapnpAgent struct {
	Provider provider.Provider
	Context  provider.ProviderParams
}

// Process implements the Agent.process method
func (a *CapnpAgent) Process(ctx context.Context, call Agent_process) error {
	// Get the input parameters
	params, err := call.Args().Params()
	if err != nil {
		return fmt.Errorf("failed to get params: %v", err)
	}
	if !params.IsValid() {
		return fmt.Errorf("invalid parameters")
	}

	// Forward to provider
	future, release := a.Provider.Complete(ctx, func(p provider.Provider_complete_Params) error {
		return p.SetParams(params)
	})
	defer release()

	// Get the provider response
	response, err := future.Struct()
	if err != nil {
		return fmt.Errorf("failed to get response: %v", err)
	}

	// Update agent context with the response
	a.Context = response

	// Allocate results
	results, err := call.AllocResults()
	if err != nil {
		return fmt.Errorf("failed to allocate results: %v", err)
	}

	// Copy response to results
	return provider.CopyProviderParams(&response, &results)
}

// GetName implements the Agent.getName method
func (a *CapnpAgent) GetName(ctx context.Context, call Agent_getName) error {
	results, err := call.AllocResults()
	if err != nil {
		return fmt.Errorf("failed to allocate results: %v", err)
	}

	return results.SetName("CapnpAgent")
}

// GetContext implements the Agent.getContext method
func (a *CapnpAgent) GetContext(ctx context.Context, call Agent_getContext) error {
	results, err := call.AllocResults()
	if err != nil {
		return fmt.Errorf("failed to allocate results: %v", err)
	}

	// Copy context to results
	return provider.CopyProviderParams(&a.Context, &results)
}

// SetContext implements the Agent.setContext method
func (a *CapnpAgent) SetContext(ctx context.Context, call Agent_setContext) error {
	params, err := call.Args().Params()
	if err != nil {
		return fmt.Errorf("failed to get params: %v", err)
	}
	if !params.IsValid() {
		return fmt.Errorf("invalid parameters")
	}

	// Create new context in the same segment
	newContext, err := provider.NewProviderParams(params.Segment())
	if err != nil {
		return fmt.Errorf("failed to create new context: %v", err)
	}

	// Copy parameters to new context
	if err := provider.CopyProviderParams(&params, &newContext); err != nil {
		return fmt.Errorf("failed to copy context: %v", err)
	}

	a.Context = newContext
	return nil
}

// AddTool implements the Agent.addTool method
func (a *CapnpAgent) AddTool(ctx context.Context, call Agent_addTool) error {
	params := call.Args()
	tool, err := params.Tool()
	if err != nil {
		return fmt.Errorf("failed to get tool: %v", err)
	}

	// Add tool to context
	tools, err := a.Context.Tools()
	if err != nil {
		return fmt.Errorf("failed to get tools: %v", err)
	}

	newTools, err := provider.NewTool_List(a.Context.Segment(), int32(tools.Len()+1))
	if err != nil {
		return fmt.Errorf("failed to create new tool list: %v", err)
	}

	// Copy existing tools
	for i := 0; i < tools.Len(); i++ {
		newTools.Set(i, tools.At(i))
	}
	// Add new tool
	newTools.Set(tools.Len(), tool)

	return a.Context.SetTools(newTools)
}

// ListTools implements the Agent.listTools method
func (a *CapnpAgent) ListTools(ctx context.Context, call Agent_listTools) error {
	results, err := call.AllocResults()
	if err != nil {
		return fmt.Errorf("failed to allocate results: %v", err)
	}

	tools, err := a.Context.Tools()
	if err != nil {
		return fmt.Errorf("failed to get tools: %v", err)
	}

	return results.SetTools(tools)
}

// NewAgent creates a new agent with the given provider
func NewAgent(prov provider.Provider) (Agent, error) {
	// Create a new message
	_, seg, err := capnp.NewMessage(capnp.SingleSegment(nil))
	if err != nil {
		return Agent{}, fmt.Errorf("failed to create message: %v", err)
	}

	// Create initial context
	context, err := provider.NewProviderParams(seg)
	if err != nil {
		return Agent{}, fmt.Errorf("failed to create context: %v", err)
	}

	return Agent_ServerToClient(&CapnpAgent{
		Provider: prov,
		Context:  context,
	}), nil
}

// Ask sends a message to the agent and returns the updated ProviderParams containing the full conversation
func Ask(ctx context.Context, agent Agent, params *provider.ProviderParams) (*provider.ProviderParams, error) {
	// Process the message through the agent
	future, release := agent.Process(ctx, func(p Agent_process_Params) error {
		return p.SetParams(*params)
	})
	defer release()

	// Get the response
	response, err := future.Struct()
	if err != nil {
		return nil, fmt.Errorf("failed to get response: %v", err)
	}

	// Create a new message and segment for the response
	_, seg, err := capnp.NewMessage(capnp.SingleSegment(nil))
	if err != nil {
		return nil, fmt.Errorf("failed to create new message: %v", err)
	}

	// Create new provider params in the new segment
	newParams, err := provider.NewRootProviderParams(seg)
	if err != nil {
		return nil, fmt.Errorf("failed to create new params: %v", err)
	}

	// Copy the response data to the new params
	if err := provider.CopyProviderParams(&response, &newParams); err != nil {
		return nil, fmt.Errorf("failed to copy response: %v", err)
	}

	// Get messages from response for logging
	messages, err := response.Messages()
	if err != nil {
		return nil, fmt.Errorf("failed to get messages: %v", err)
	}

	// Copy messages to new params
	if messages.Len() > 0 {
		newMessages, err := newParams.NewMessages(int32(messages.Len()))
		if err != nil {
			return nil, fmt.Errorf("failed to create new messages: %v", err)
		}

		for i := 0; i < messages.Len(); i++ {
			newMessages.Set(i, messages.At(i))
		}
	}

	return &newParams, nil
}
