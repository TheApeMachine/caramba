package agent

import (
	"context"
	"io"
	"sync"
	"time"

	"capnproto.org/go/capnp/v3/rpc"
	"github.com/theapemachine/caramba/pkg/datura"
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
