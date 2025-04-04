package agent

import (
	"context"
	"sync"

	"capnproto.org/go/capnp/v3"
)

// StateServer manages the state for a single agent
type StateServer struct {
	mu    sync.RWMutex
	agent Agent // The agent's current state
}

// NewStateServer creates a new state server for an agent
func NewStateServer(segment *capnp.Segment) (*StateServer, error) {
	agent, err := NewAgent(segment)
	if err != nil {
		return nil, err
	}

	return &StateServer{
		agent: agent,
	}, nil
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

	srv.mu.RLock()
	defer srv.mu.RUnlock()

	// Get current state
	params, err := srv.agent.Params()
	if err != nil {
		return err
	}

	context, err := srv.agent.Context()
	if err != nil {
		return err
	}

	// Return current state
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

	srv.mu.Lock()
	defer srv.mu.Unlock()

	// Update agent state
	if err := srv.agent.SetParams(params); err != nil {
		return err
	}

	if err := srv.agent.SetContext(context); err != nil {
		return err
	}

	return nil
}
