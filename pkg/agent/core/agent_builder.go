package core

import "github.com/theapemachine/caramba/pkg/process"

// AgentBuilder provides a fluent interface for building agents
type AgentBuilder struct {
	agent *BaseAgent
}

// NewAgentBuilder creates a new AgentBuilder
func NewAgentBuilder(name string) *AgentBuilder {
	return &AgentBuilder{
		agent: NewBaseAgent(name),
	}
}

// WithLLM sets the LLM provider for the agent
func (b *AgentBuilder) WithLLM(llm LLMProvider) *AgentBuilder {
	b.agent.SetLLM(llm)
	return b
}

// WithSystemPrompt sets the system prompt for the agent
func (b *AgentBuilder) WithSystemPrompt(prompt string) *AgentBuilder {
	b.agent.SetSystemPrompt(prompt)
	return b
}

// WithIterationLimit sets the iteration limit for the agent
func (b *AgentBuilder) WithIterationLimit(limit int) *AgentBuilder {
	b.agent.SetIterationLimit(limit)
	return b
}

// WithMemory sets the memory system for the agent
func (b *AgentBuilder) WithMemory(memory Memory) *AgentBuilder {
	b.agent.SetMemory(memory)
	return b
}

// WithPlanner sets the planner for the agent
func (b *AgentBuilder) WithPlanner(planner Agent) *AgentBuilder {
	b.agent.SetPlanner(planner)
	return b
}

// WithOptimizer sets the optimizer for the agent
func (b *AgentBuilder) WithOptimizer(optimizer Agent) *AgentBuilder {
	b.agent.SetOptimizer(optimizer)
	return b
}

// WithTool adds a tool to the agent
func (b *AgentBuilder) WithTool(tool Tool) *AgentBuilder {
	b.agent.AddTool(tool)
	return b
}

// WithProcess sets the process for the agent
func (b *AgentBuilder) WithProcess(process process.StructuredOutput) *AgentBuilder {
	b.agent.SetProcess(process)
	return b
}

// WithStreaming sets the streaming mode for the agent
func (b *AgentBuilder) WithStreaming(streaming bool) *AgentBuilder {
	b.agent.SetStreaming(streaming)
	return b
}

// Build builds and returns the agent
func (b *AgentBuilder) Build() Agent {
	return b.agent
}
