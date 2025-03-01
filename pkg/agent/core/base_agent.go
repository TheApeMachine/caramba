package core

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
	"github.com/theapemachine/caramba/pkg/process"
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
	params         *LLMParams
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
		params: &LLMParams{
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
				if err = agent.handleEvent(ctx, event); err != nil {
					agent.logger.Log(fmt.Sprintf("Error handling event: %s", err))
					return
				}
			default:
				time.Sleep(1 * time.Second)
			}
		}
	}()

	return out, err
}

func (agent *BaseAgent) handleEvent(ctx context.Context, event *hub.Event) (err error) {
	// Convert the event to an LLM message
	message := LLMMessage{
		Role:    event.Role,
		Content: event.Message,
	}

	agent.logger.Log(fmt.Sprintf("Received event: %s", event.String()))

	// Define our agent chain
	for _, active := range []Agent{agent.Planner, agent, agent.Optimizer} {
		if active == nil {
			continue
		}

		if message, err = NewIterationManager(active).Run(ctx, message); err != nil {
			agent.hub.Add(hub.NewEvent(
				agent.name,
				"error",
				"agent",
				hub.EventTypeError,
				err.Error(),
				map[string]string{},
			))
			return err
		}
	}

	return nil
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

func (agent *BaseAgent) Streaming() bool {
	return agent.streaming
}

func (agent *BaseAgent) Params() *LLMParams {
	return agent.params
}

func (agent *BaseAgent) Status() AgentStatus {
	return agent.status
}

func (agent *BaseAgent) SetStatus(status AgentStatus) {
	agent.status = status
}

// AddTool adds a new tool to the agent.
func (agent *BaseAgent) AddTool(tool Tool) {
	agent.params.Tools = append(agent.params.Tools, tool)
}

func (agent *BaseAgent) AddUserMessage(message string) {
	agent.params.Messages = append(agent.params.Messages, LLMMessage{
		Role:    "user",
		Content: message,
	})
}

func (agent *BaseAgent) AddAssistantMessage(message string) {
	agent.params.Messages = append(agent.params.Messages, LLMMessage{
		Role:    "assistant",
		Content: message,
	})
}

// SetMemory sets the memory system for the agent.
func (agent *BaseAgent) SetMemory(memory Memory) {
	agent.memory = memory
}

// SetPlanner sets the planner for the agent.
func (agent *BaseAgent) SetPlanner(planner Agent) {
	agent.Planner = planner
}

// SetOptimizer sets the optimizer for the agent.
func (agent *BaseAgent) SetOptimizer(optimizer Agent) {
	agent.Optimizer = optimizer
}

// SetLLM sets the LLM provider for the agent.
func (agent *BaseAgent) SetLLM(llm LLMProvider) {
	agent.llm = llm
}

// SetSystemPrompt sets the system prompt for the agent.
func (agent *BaseAgent) SetSystemPrompt(prompt string) {
	agent.params.Messages = append(agent.params.Messages, SystemMessage(prompt))
}

// SetProcess sets the process for the agent.
func (agent *BaseAgent) SetProcess(process process.StructuredOutput) {
	agent.params.ResponseFormatName = process.Name()
	agent.params.ResponseFormatDescription = process.Description()
	agent.params.Schema = process.Schema()
}

// SetModel sets the model for the agent.
func (agent *BaseAgent) SetModel(model string) {
	agent.params.Model = model
}

// SetTemperature sets the temperature for the agent.
func (agent *BaseAgent) SetTemperature(temperature float64) {
	agent.params.Temperature = temperature
}

// SetIterationLimit sets the iteration limit for the agent.
func (agent *BaseAgent) SetIterationLimit(limit int) {
	agent.iterationLimit = limit
}

// GetMessenger returns the agent's messenger.
func (agent *BaseAgent) GetMessenger() Messenger {
	return agent.Messenger
}

// SetMessenger sets the agent's messenger.
func (agent *BaseAgent) SetMessenger(messenger Messenger) {
	agent.Messenger = messenger
}

// SetStreaming sets the streaming mode for the agent.
func (agent *BaseAgent) SetStreaming(streaming bool) {
	agent.streaming = streaming
}

func (agent *BaseAgent) GetTool(name string) Tool {
	for _, tool := range agent.params.Tools {
		if tool.Name() == name {
			return tool
		}
	}

	return nil
}
