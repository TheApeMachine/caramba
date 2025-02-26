package core

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

// WithTool adds a tool to the agent
func (b *AgentBuilder) WithTool(tool Tool) *AgentBuilder {
	_ = b.agent.AddTool(tool)
	return b
}

// WithPlanner sets the planner for the agent
func (b *AgentBuilder) WithPlanner(planner Planner) *AgentBuilder {
	b.agent.SetPlanner(planner)
	return b
}

// WithMessenger sets the messenger for the agent
func (b *AgentBuilder) WithMessenger(messenger Messenger) *AgentBuilder {
	b.agent.Messenger = messenger
	return b
}

// Build builds and returns the agent
func (b *AgentBuilder) Build() Agent {
	return b.agent
}
