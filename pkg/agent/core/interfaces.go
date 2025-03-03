package core

import (
	"context"

	"github.com/theapemachine/caramba/pkg/process"
)

/*
Agent represents the interface for an LLM-powered agent.
It defines the core functionality for executing tasks, managing tools,
memory, planning, and communication with other agents.
*/
type Agent interface {
	Execute(context.Context) (string, error)
	Name() string
	IterationLimit() int
	Params() *LLMParams
	SystemPrompt() string
	SetStatus(status AgentStatus)
	Status() AgentStatus
	AddTools(tools ...Tool)
	AddUserMessage(message string)
	AddAssistantMessage(message string)
	SetParent(parent Agent)
	Parent() Agent
	SetMemory(memory Memory)
	Memory() Memory
	SetPlanner(planner Agent)
	SetOptimizer(optimizer Agent)
	SetLLM(llm LLMProvider)
	LLM() LLMProvider
	SetSystemPrompt(prompt string)
	SetProcess(process process.StructuredOutput)
	SetSubscriptions(subscriptions ...string)
	SetIterationLimit(limit int)
	GetTool(name string) Tool
	SetStreaming(streaming bool)
	Streaming() bool
}

/*
LLMProvider defines the interface for language model providers.
It abstracts the interaction with various LLM services and allows
for a consistent API regardless of the underlying provider.
*/
type LLMProvider interface {
	GenerateResponse(context.Context, LLMParams) LLMResponse
	StreamResponse(context.Context, LLMParams) <-chan LLMResponse
	Name() string
}

type LLMMessage struct {
	Role    string
	Content string
}

func SystemMessage(content string) LLMMessage {
	return LLMMessage{
		Role:    "system",
		Content: content,
	}
}

type LLMParams struct {
	Messages                  []LLMMessage
	Model                     string
	MaxTokens                 int
	TopP                      float64
	Temperature               float64
	FrequencyPenalty          float64
	PresencePenalty           float64
	StopSequences             []string
	SystemPrompt              string
	Tools                     []Tool
	ResponseFormatName        string
	ResponseFormatDescription string
	Schema                    process.StructuredOutput
}

type ResponseType string

const (
	ResponseTypeContent  ResponseType = "content"
	ResponseTypeToolCall ResponseType = "tool_call"
	ResponseTypeError    ResponseType = "error"
)

type LLMResponse struct {
	Type      ResponseType
	Model     string
	Content   string
	ToolCalls []ToolCall
	Refusal   string
	Error     error
}

type ToolCall struct {
	Name string
	Args map[string]any
}

/*
Memory defines the interface for the agent's memory system.
It provides methods for storing, retrieving, and searching information
that the agent needs to persist across interactions.
*/
type Memory interface {
	SetParent(parent Agent)
	QueryAgent() Agent
	MutateAgent() Agent
	Query(context.Context, *process.MemoryLookup) (string, error)
	Mutate(context.Context, *process.MemoryMutate) error
}

/*
MemoryEnhancer defines the interface for memory systems that can enhance prompts.
It provides methods to prepare context with relevant memories for a given query.
*/
type MemoryEnhancer interface {
	PrepareContext(ctx context.Context, agentID string, query string) (string, error)
}

/*
MemoryExtractor defines the interface for memory systems that can extract memories.
It provides methods to extract important information from text to be stored as memories.
*/
type MemoryExtractor interface {
	ExtractMemories(ctx context.Context, agentID string, text string, source string) ([]string, error)
}

/*
MemoryEntry represents an entry in the memory system.
It includes the key, value, and a relevance score for search results.
*/
type MemoryEntry struct {
	Key   string
	Value any
	Score float64
}

/*
Tool defines the interface for agent tools.
Tools extend an agent's capabilities by allowing it to perform
specific actions like web searches, calculations, or API calls.
*/
type Tool interface {
	Name() string
	Description() string
	Execute(ctx context.Context, args map[string]any) (any, error)
	Schema() map[string]any
}

/*
Planner defines the interface for the agent's planning component.
It enables the agent to break down complex tasks into manageable steps
and execute them in sequence.
*/
type Planner interface {
	CreatePlan(ctx context.Context, input string) (Plan, error)
	ExecutePlan(ctx context.Context, plan Plan) (string, error)
	GuideAgent(ctx context.Context, agent Agent, plan Plan, query string) (string, error)
}

/*
Plan represents a plan created by a planner.
It consists of a sequence of steps to be executed.
*/
type Plan struct {
	Steps []PlanStep
}

/*
PlanStep represents a step in a plan.
It includes a description of the step, the tool to use, and the arguments for that tool.
*/
type PlanStep struct {
	Description string
	ToolName    string
	Arguments   map[string]any
}

/*
Workflow defines the interface for a workflow.
Workflows allow for creating sequences of tool executions with
conditional logic and error handling.
*/
type Workflow interface {
	AddStep(name string, tool Tool, args map[string]any) Workflow
	AddConditionalStep(name string, condition string, tool Tool, args map[string]any) Workflow
	Execute(ctx context.Context, input map[string]any) (map[string]any, error)
	SetErrorHandler(handler func(error) error) Workflow
}
