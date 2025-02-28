package core

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
)

type AgentStatus string

const (
	AgentStatusIdle     AgentStatus = "idle"
	AgentStatusRunning  AgentStatus = "running"
	AgentStatusThinking AgentStatus = "thinking"
	AgentStatusError    AgentStatus = "error"
	AgentStatusSuccess  AgentStatus = "success"
)

// BaseAgent provides a base implementation of the Agent interface.
type BaseAgent struct {
	logger         *output.Logger
	hub            *hub.Queue
	name           string
	memory         Memory
	llm            LLMProvider
	Params         *LLMParams
	Planner        Agent
	Optimizer      Agent
	Messenger      Messenger
	toolsMu        sync.RWMutex
	iterationLimit int
	streaming      bool
	status         AgentStatus
}

// NewBaseAgent creates a new BaseAgent.
func NewBaseAgent(name string) *BaseAgent {
	return &BaseAgent{
		logger: output.NewLogger(),
		hub:    hub.NewQueue(),
		name:   name,
		Params: &LLMParams{
			Messages:                  make([]LLMMessage, 0),
			Model:                     "gpt-4o-mini",
			Temperature:               0.7,
			MaxTokens:                 1024,
			Tools:                     make([]Tool, 0),
			ResponseFormatName:        "",
			ResponseFormatDescription: "",
			Schema:                    nil,
		},
		iterationLimit: 1,
		streaming:      false,
		Messenger:      NewInMemoryMessenger(name),
		status:         AgentStatusIdle,
	}
}

// Execute runs the agent with the provided input and returns a response (non-streaming).
func (agent *BaseAgent) Execute(ctx context.Context) (out string, err error) {
	agent.logger.Log(fmt.Sprintf("Executing agent %s", agent.name))
	events := agent.hub.Subscribe(agent.name)

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case event := <-events:
				// Convert the event to a LLMMessage
				message := LLMMessage{
					Role:    event.Role,
					Content: event.Message,
				}

				agent.logger.Log(fmt.Sprintf("Received event: %s", event.String()))

				if agent.Planner != nil {
					if err = NewIterationManager(agent.Planner).Run(
						ctx, message, agent.Params, agent.streaming,
					); err != nil {
						// If the planner fails, the agent is still able to continue, it will just
						// be in a degraded state.
						agent.hub.Add(hub.NewEvent(
							agent.name,
							"error",
							"planner",
							hub.EventTypeError,
							err.Error(),
							map[string]string{},
						))
					}
				}

				if err = NewIterationManager(agent).Run(ctx, message, agent.Params, agent.streaming); err != nil {
					// If the agent fails, there is no point in continuing.
					agent.hub.Add(hub.NewEvent(
						agent.name,
						"error",
						"agent",
						hub.EventTypeError,
						err.Error(),
						map[string]string{},
					))
					return
				}

				if agent.Optimizer != nil {
					if err = NewIterationManager(agent.Optimizer).Run(
						ctx, message, agent.Params, agent.streaming,
					); err != nil {
						// If the optimizer fails, the agent is still able to continue, it will just
						// not self-optimize.
						agent.hub.Add(hub.NewEvent(
							agent.name,
							"error",
							"optimizer",
							hub.EventTypeError,
							err.Error(),
							map[string]string{},
						))
					}
				}
			default:
				time.Sleep(1 * time.Second)
			}
		}
	}()

	return out, err
}

func (agent *BaseAgent) Name() string {
	return agent.name
}

func (agent *BaseAgent) Memory() Memory {
	return agent.memory
}

func (agent *BaseAgent) IterationLimit() int {
	return agent.iterationLimit
}

func (agent *BaseAgent) LLM() LLMProvider {
	return agent.llm
}

func (agent *BaseAgent) Status() AgentStatus {
	return agent.status
}

func (agent *BaseAgent) SetStatus(status AgentStatus) {
	agent.status = status
}

// AddTool adds a new tool to the agent.
func (agent *BaseAgent) AddTool(tool Tool) error {
	agent.toolsMu.Lock()
	defer agent.toolsMu.Unlock()

	for _, existingTool := range agent.Params.Tools {
		if existingTool.Name() == tool.Name() {
			return fmt.Errorf("tool %s already exists", tool.Name())
		}
	}
	agent.Params.Tools = append(agent.Params.Tools, tool)
	agent.hub.Add(hub.NewEvent(
		agent.name,
		"info",
		"tool",
		hub.EventTypeInfo,
		fmt.Sprintf("Added tool %s to agent %s", tool.Name(), agent.name),
		map[string]string{},
	))
	return nil
}

// SetMemory sets the memory system for the agent.
func (agent *BaseAgent) SetMemory(memory Memory) {
	agent.memory = memory
	output.Verbose(fmt.Sprintf("Set memory system for agent %s", agent.name))
}

// SetPlanner sets the planner for the agent.
func (agent *BaseAgent) SetPlanner(planner Agent) {
	agent.Planner = planner
	output.Verbose(fmt.Sprintf("Set planner for agent %s", agent.name))
}

// SetOptimizer sets the optimizer for the agent.
func (agent *BaseAgent) SetOptimizer(optimizer Agent) {
	agent.Optimizer = optimizer
	output.Verbose(fmt.Sprintf("Set optimizer for agent %s", agent.name))
}

// SetLLM sets the LLM provider for the agent.
func (agent *BaseAgent) SetLLM(llm LLMProvider) {
	agent.llm = llm
	output.Verbose(fmt.Sprintf("Set LLM provider %s for agent %s", llm.Name(), agent.name))
}

// SetSystemPrompt sets the system prompt for the agent.
func (agent *BaseAgent) SetSystemPrompt(prompt string) {
	agent.Params.Messages = append(agent.Params.Messages, SystemMessage(prompt))
	output.Verbose(fmt.Sprintf("Set system prompt for agent %s (%d chars)", agent.name, len(prompt)))
}

// SetModel sets the model for the agent.
func (agent *BaseAgent) SetModel(model string) {
	agent.Params.Model = model
	output.Verbose(fmt.Sprintf("Set model %s for agent %s", model, agent.name))
}

// SetTemperature sets the temperature for the agent.
func (agent *BaseAgent) SetTemperature(temperature float64) {
	agent.Params.Temperature = temperature
	output.Verbose(fmt.Sprintf("Set temperature %.2f for agent %s", temperature, agent.name))
}

// SetIterationLimit sets the iteration limit for the agent.
func (agent *BaseAgent) SetIterationLimit(limit int) {
	agent.iterationLimit = limit
	output.Verbose(fmt.Sprintf("Set iteration limit %d for agent %s", limit, agent.name))
}

// GetMessenger returns the agent's messenger.
func (agent *BaseAgent) GetMessenger() Messenger {
	return agent.Messenger
}

// SetMessenger sets the agent's messenger.
func (agent *BaseAgent) SetMessenger(messenger Messenger) {
	agent.Messenger = messenger
	output.Verbose(fmt.Sprintf("Set messenger for agent %s", agent.name))
}

// SetStreaming sets the streaming mode for the agent.
func (agent *BaseAgent) SetStreaming(streaming bool) {
	agent.streaming = streaming
	output.Verbose(fmt.Sprintf("Set streaming mode for agent %s", agent.name))
}
