package core

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/theapemachine/caramba/pkg/output"
)

// BaseAgent provides a base implementation of the Agent interface.
type BaseAgent struct {
	AgentName      string
	LongTermMemory Memory
	llm            LLMProvider
	Params         *LLMParams
	Planner        Agent
	Optimizer      Agent
	Messenger      Messenger
	toolsMu        sync.RWMutex
	iterationLimit int
	streaming      bool
}

// NewBaseAgent creates a new BaseAgent.
func NewBaseAgent(name string) *BaseAgent {
	return &BaseAgent{
		AgentName: name,
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
	}
}

// Execute runs the agent with the provided input and returns a response (non-streaming).
func (agent *BaseAgent) Execute(ctx context.Context, message LLMMessage) (out string, err error) {
	if agent.llm == nil {
		output.Error("No LLM provider set", errors.New("missing LLM"))
		return "", errors.New("missing LLM")
	}

	if agent.Planner != nil {
		if err = NewIterationManager(agent.Planner).Run(
			ctx, message, agent.Params, agent.streaming,
		); err != nil {
			// If the planner fails, the agent is still able to continue, it will just
			// be in a degraded state.
			output.Error("Planner failed", err)
		}
	}

	if err = NewIterationManager(agent).Run(ctx, message, agent.Params, agent.streaming); err != nil {
		// If the agent fails, there is no point in continuing.
		output.Error("Agent failed", err)
		return "", err
	}

	if agent.Optimizer != nil {
		if err = NewIterationManager(agent.Optimizer).Run(
			ctx, message, agent.Params, agent.streaming,
		); err != nil {
			// If the optimizer fails, the agent is still able to continue, it will just
			// not self-optimize.
			output.Error("Optimizer failed", err)
		}
	}

	return out, err
}

func (agent *BaseAgent) Name() string {
	return agent.AgentName
}

func (agent *BaseAgent) Memory() Memory {
	return agent.LongTermMemory
}

func (agent *BaseAgent) IterationLimit() int {
	return agent.iterationLimit
}

func (agent *BaseAgent) LLM() LLMProvider {
	return agent.llm
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
	output.Verbose(fmt.Sprintf("Added tool %s to agent %s", tool.Name(), agent.AgentName))
	return nil
}

// SetMemory sets the memory system for the agent.
func (agent *BaseAgent) SetMemory(memory Memory) {
	agent.LongTermMemory = memory
	output.Verbose(fmt.Sprintf("Set memory system for agent %s", agent.AgentName))
}

// SetPlanner sets the planner for the agent.
func (agent *BaseAgent) SetPlanner(planner Agent) {
	agent.Planner = planner
	output.Verbose(fmt.Sprintf("Set planner for agent %s", agent.AgentName))
}

// SetOptimizer sets the optimizer for the agent.
func (agent *BaseAgent) SetOptimizer(optimizer Agent) {
	agent.Optimizer = optimizer
	output.Verbose(fmt.Sprintf("Set optimizer for agent %s", agent.AgentName))
}

// SetLLM sets the LLM provider for the agent.
func (agent *BaseAgent) SetLLM(llm LLMProvider) {
	agent.llm = llm
	output.Verbose(fmt.Sprintf("Set LLM provider %s for agent %s", llm.Name(), agent.AgentName))
}

// SetSystemPrompt sets the system prompt for the agent.
func (agent *BaseAgent) SetSystemPrompt(prompt string) {
	agent.Params.Messages = append(agent.Params.Messages, SystemMessage(prompt))
	output.Verbose(fmt.Sprintf("Set system prompt for agent %s (%d chars)", agent.AgentName, len(prompt)))
}

// SetModel sets the model for the agent.
func (agent *BaseAgent) SetModel(model string) {
	agent.Params.Model = model
	output.Verbose(fmt.Sprintf("Set model %s for agent %s", model, agent.AgentName))
}

// SetTemperature sets the temperature for the agent.
func (agent *BaseAgent) SetTemperature(temperature float64) {
	agent.Params.Temperature = temperature
	output.Verbose(fmt.Sprintf("Set temperature %.2f for agent %s", temperature, agent.AgentName))
}

// SetIterationLimit sets the iteration limit for the agent.
func (agent *BaseAgent) SetIterationLimit(limit int) {
	agent.iterationLimit = limit
	output.Verbose(fmt.Sprintf("Set iteration limit %d for agent %s", limit, agent.AgentName))
}

// GetMessenger returns the agent's messenger.
func (agent *BaseAgent) GetMessenger() Messenger {
	return agent.Messenger
}

// SetMessenger sets the agent's messenger.
func (agent *BaseAgent) SetMessenger(messenger Messenger) {
	agent.Messenger = messenger
	output.Verbose(fmt.Sprintf("Set messenger for agent %s", agent.AgentName))
}

// SetStreaming sets the streaming mode for the agent.
func (agent *BaseAgent) SetStreaming(streaming bool) {
	agent.streaming = streaming
	output.Verbose(fmt.Sprintf("Set streaming mode for agent %s", agent.AgentName))
}
