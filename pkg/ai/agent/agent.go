package agent

import (
	"context"
	"io"
	"sync"
	"time"

	"capnproto.org/go/capnp/v3/rpc"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

// AgentServer implements the AgentRPC interface
type AgentServer struct {
	agent    *Agent
	ctx      context.Context
	cancel   context.CancelFunc
	conn     *rpc.Conn
	mu       sync.RWMutex
	handlers map[string]chan *datura.ArtifactBuilder
	msgCh    chan *datura.ArtifactBuilder
}

// NewAgentServer creates a new AgentServer
func NewAgentServer(agent *Agent) *AgentServer {
	ctx, cancel := context.WithCancel(context.Background())
	return &AgentServer{
		agent:    agent,
		ctx:      ctx,
		cancel:   cancel,
		conn:     nil,
		handlers: make(map[string]chan *datura.ArtifactBuilder),
		msgCh:    make(chan *datura.ArtifactBuilder, 10),
	}
}

// Run starts the agent's main loop
func (srv *AgentServer) Run(ctx context.Context, call AgentRPC_run) error {
	// Create a ticker for periodic state checks
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case msg := <-srv.msgCh:
			if err := srv.handleMessage(ctx, msg); err != nil {
				return err
			}
		case <-ticker.C:
			// Process any pending messages
			srv.mu.RLock()
			for topic, ch := range srv.handlers {
				select {
				case msg := <-ch:
					if err := srv.processMessage(ctx, topic, msg); err != nil {
						srv.mu.RUnlock()
						return err
					}
				default:
					// No messages waiting
				}
			}
			srv.mu.RUnlock()

			// Check agent state and perform any needed actions
			if err := srv.checkState(ctx); err != nil {
				return err
			}
		}
	}
}

// Stop stops the agent
func (srv *AgentServer) Stop(ctx context.Context, call AgentRPC_stop) error {
	srv.cancel()
	return nil
}

// Run starts the agent with the given RPC connection
func (agent *Agent) Run(ctx context.Context, transport io.ReadWriteCloser) *Agent {
	server := NewAgentServer(agent)
	server.conn = agent.Conn(transport)

	if err := server.Run(ctx, AgentRPC_run{}); err != nil {
		return nil
	}

	return agent
}

// AgentToClient converts an Agent to a client capability
func AgentToClient(agent *Agent) AgentRPC {
	server := NewAgentServer(agent)
	return AgentRPC_ServerToClient(server)
}

// handleMessage processes an incoming message
func (srv *AgentServer) handleMessage(ctx context.Context, msg *datura.ArtifactBuilder) error {
	// Get the message topic
	topic := datura.GetMetaValue[string](msg, "topic")
	if topic == "" {
		return errnie.Error("message has no topic")
	}

	// Get or create the handler channel for this topic
	srv.mu.Lock()
	ch, exists := srv.handlers[topic]
	if !exists {
		ch = make(chan *datura.ArtifactBuilder, 10)
		srv.handlers[topic] = ch
	}
	srv.mu.Unlock()

	// Send the message to the handler
	select {
	case ch <- msg:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// processMessage handles messages for specific topics
func (srv *AgentServer) processMessage(ctx context.Context, topic string, msg *datura.ArtifactBuilder) error {
	switch topic {
	case "command":
		// Process command messages
		if err := srv.agent.ProcessCommand(msg); err != nil {
			return errnie.Error(err)
		}
	case "status":
		// Process status update messages
		if err := srv.agent.UpdateStatus(msg); err != nil {
			return errnie.Error(err)
		}
	default:
		// Process other messages
		if err := srv.agent.ProcessMessage(msg); err != nil {
			return errnie.Error(err)
		}
	}
	return nil
}

// checkState performs periodic state checks and maintenance
func (srv *AgentServer) checkState(ctx context.Context) error {
	// Check if the agent is still active
	if !srv.agent.IsActive() {
		return errnie.Error("agent is no longer active")
	}

	// Perform maintenance tasks
	if err := srv.agent.Maintain(); err != nil {
		return errnie.Error(err)
	}

	return nil
}

// ProcessCommand processes a command message
func (a *Agent) ProcessCommand(msg *datura.ArtifactBuilder) error {
	// Get the command from the message
	cmd := datura.GetMetaValue[string](msg, "command")
	if cmd == "" {
		return errnie.Error("command is empty")
	}

	// Process the command based on its type
	switch cmd {
	case "start":
		// Handle start command
		return nil
	case "stop":
		// Handle stop command
		return nil
	default:
		return errnie.Error("unknown command")
	}
}

// UpdateStatus updates the agent's status based on a status message
func (a *Agent) UpdateStatus(msg *datura.ArtifactBuilder) error {
	// Get the status from the message
	status := datura.GetMetaValue[string](msg, "status")
	if status == "" {
		return errnie.Error("status is empty")
	}

	// Update the agent's status
	// TODO: Implement status update logic
	return nil
}

// ProcessMessage processes a general message
func (a *Agent) ProcessMessage(msg *datura.ArtifactBuilder) error {
	// Process the message based on its content
	// TODO: Implement message processing logic
	return nil
}

// IsActive checks if the agent is still active
func (a *Agent) IsActive() bool {
	// TODO: Implement proper activity check
	return true
}

// Maintain performs periodic maintenance tasks
func (a *Agent) Maintain() error {
	// TODO: Implement maintenance tasks
	return nil
}
