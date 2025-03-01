package core

import (
	"context"
	"fmt"
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
	iterationLimit int
	streaming      bool
	status         AgentStatus
}

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
	message := LLMMessage{
		Role:    event.Role,
		Content: event.Message,
	}

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

func (agent *BaseAgent) SetMemory(memory Memory) {
	agent.memory = memory
}

func (agent *BaseAgent) SetPlanner(planner Agent) {
	agent.Planner = planner
}

func (agent *BaseAgent) SetOptimizer(optimizer Agent) {
	agent.Optimizer = optimizer
}

func (agent *BaseAgent) SetLLM(llm LLMProvider) {
	agent.llm = llm
}

func (agent *BaseAgent) SetSystemPrompt(prompt string) {
	agent.params.Messages = append(agent.params.Messages, SystemMessage(prompt))
}

func (agent *BaseAgent) SetProcess(process process.StructuredOutput) {
	agent.params.ResponseFormatName = process.Name()
	agent.params.ResponseFormatDescription = process.Description()
	agent.params.Schema = process.Schema()
}

func (agent *BaseAgent) SetModel(model string) {
	agent.params.Model = model
}

func (agent *BaseAgent) SetTemperature(temperature float64) {
	agent.params.Temperature = temperature
}

func (agent *BaseAgent) SetIterationLimit(limit int) {
	agent.iterationLimit = limit
}

func (agent *BaseAgent) GetMessenger() Messenger {
	return agent.Messenger
}

func (agent *BaseAgent) SetMessenger(messenger Messenger) {
	agent.Messenger = messenger
}

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
