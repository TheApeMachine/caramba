package agent

import (
	"context"
)

// GetState returns the agent's current state
func (srv *AgentServer) GetState(ctx context.Context, call AgentRPC_getState) error {
	results, err := call.AllocResults()
	if err != nil {
		return err
	}

	return results.SetState(*srv.agent)
}

// SetState updates the agent's state
func (srv *AgentServer) SetState(ctx context.Context, call AgentRPC_setState) error {
	state, err := call.Args().State()
	if err != nil {
		return err
	}

	// Update agent state
	*srv.agent = state
	return nil
}

// checkState verifies the agent's state and performs any necessary actions
func (srv *AgentServer) checkState(ctx context.Context) error {
	// TODO: Implement state checking logic
	return nil
}

// StateServer manages the state for a single agent
type StateServer struct {
	agent *Agent // The agent's current state
}

type StateServerOption func(*StateServer) error

// NewStateServer creates a new state server for an agent
func NewStateServer(options ...StateServerOption) (*StateServer, error) {
	var (
		srv = &StateServer{}
	)

	for _, option := range options {
		if err := option(srv); err != nil {
			return nil, err
		}
	}

	return srv, nil
}

// Get returns the agent's current parameters and context
func (srv *StateServer) Get(
	ctx context.Context,
	call State_get,
) error {
	results, err := call.AllocResults()

	if err != nil {
		return err
	}

	// Get current state
	params, err := srv.agent.Params()

	if err != nil {
		return err
	}

	context, err := srv.agent.Context()

	if err != nil {
		return err
	}

	if err := results.SetParams(params); err != nil {
		return err
	}

	if err := results.SetContext(context); err != nil {
		return err
	}

	return nil
}

// Set updates the agent's parameters and context
func (srv *StateServer) Set(
	ctx context.Context,
	call State_set,
) error {
	args := call.Args()

	// Get new state from arguments
	params, err := args.Params()
	if err != nil {
		return err
	}

	context, err := args.Context()
	if err != nil {
		return err
	}

	// Update agent state
	if err := srv.agent.SetParams(params); err != nil {
		return err
	}

	if err := srv.agent.SetContext(context); err != nil {
		return err
	}

	return nil
}

func WithAgent(agent *Agent) StateServerOption {
	return func(srv *StateServer) error {
		srv.agent = agent
		return nil
	}
}
